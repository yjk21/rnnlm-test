///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.4a
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
// (c) 2013 Cantab Research Ltd (info@cantabResearch.com)
//
///////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <limits.h>
#include "rnnlmlib.h"

using namespace std;

int argPos(char *str, int argc, char **argv)
{
    int a;
    for (a=1; a<argc; a++) if (!strcmp(str, argv[a])) return a;
    return -1;
}

class CRnnLMaccess : public CRnnLM{
    public:

        CRnnLMaccess():CRnnLM(){}

        void trainNet()
        {
            //This was copied from rnnlmlib.cpp to include statements to save the initial state
            int a, b, word, last_word, wordcn;
            char log_name[200];
            FILE *fi, *flog;
            clock_t start, now;

            sprintf(log_name, "%s.output.txt", rnnlm_file);

            printf("Starting training using file %s\n", train_file);
            starting_alpha=alpha;

            fi=fopen(rnnlm_file, "rb");
            if (fi!=NULL) {
                fclose(fi);
                printf("Restoring network from file to continue training...\n");
                restoreNet();
            } else {
                learnVocabFromTrainFile();
                initNet();
                iter=0;
            }

            dumpVocab();
            dumpSynapses();

            if (class_size>vocab_size) {
                printf("WARNING: number of classes exceeds vocabulary size!\n");
            }

            counter=train_cur_pos;
            //saveNet();
            while (iter < maxIter) {
                printf("Iter: %3d\tAlpha: %f\t   ", iter, alpha);
                fflush(stdout);

                if (bptt>0) for (a=0; a<bptt+bptt_block; a++) bptt_history[a]=0;
                for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;

                //TRAINING PHASE
                netFlush();

                fi=fopen(train_file, "rb");
                last_word=0;

                if (counter>0) for (a=0; a<counter; a++) word=readWordIndex(fi);	//this will skip words that were already learned if the training was interrupted

                start=clock();

                while (1) {
                    counter++;

                    if ((counter%10000)==0) if ((debug_mode>1)) {
                        now=clock();
                        if (train_words>0)
                            printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %.2f%%   Words/sec: %.1f ", 13, iter, alpha, -logp/counter, counter/(real)train_words*100, counter/((double)(now-start)/1000000.0));
                        else
                            printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %dK", 13, iter, alpha, -logp/counter, counter/1000);
                        fflush(stdout);
                    }

                    if ((anti_k>0) && ((counter%anti_k)==0)) {
                        train_cur_pos=counter;
                        saveNet();
                    }

                    word=readWordIndex(fi);     //read next word
                    computeNet(last_word, word);      //compute probability distribution
                    if (feof(fi)) break;        //end of file: test on validation data, iterate till convergence

                    if (word!=-1) logp+=log(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac);

                    if ((logp!=logp) || (isinf(logp))) {
                        printf("\nNumerical error %d %f %f\n", word, neu2[word].ac, neu2[vocab[word].class_index+vocab_size].ac);
                        exit(1);
                    }

                    //
                    if (bptt>0) {		//shift memory needed for bptt to next time step
                        for (a=bptt+bptt_block-1; a>0; a--) bptt_history[a]=bptt_history[a-1];
                        bptt_history[0]=last_word;

                        for (a=bptt+bptt_block-1; a>0; a--) for (b=0; b<layer1_size; b++) {
                            bptt_hidden[a*layer1_size+b].ac=bptt_hidden[(a-1)*layer1_size+b].ac;
                            bptt_hidden[a*layer1_size+b].er=bptt_hidden[(a-1)*layer1_size+b].er;
                        }
                    }
                    //
                    learnNet(last_word, word);

                    copyHiddenLayerToInput();

                    if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation

                    last_word=word;

                    for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
                    history[0]=last_word;

                    if (independent && (word==0)) netReset();
                }
                fclose(fi);

                now=clock();
                printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Words/sec: %.1f   ", 13, iter, alpha, -logp/counter, counter/((double)(now-start)/1000000.0));

                if (one_iter==1) {	//no validation data are needed and network is always saved with modified weights
                    printf("\n");
                    logp=0;
                    saveNet();
                    break;
                }

                //VALIDATION PHASE
                netFlush();

                fi=fopen(valid_file, "rb");
                if (fi==NULL) {
                    printf("Valid file not found\n");
                    exit(1);
                }

                flog=fopen(log_name, "ab");
                if (flog==NULL) {
                    printf("Cannot open log file\n");
                    exit(1);
                }

                //fprintf(flog, "Index   P(NET)          Word\n");
                //fprintf(flog, "----------------------------------\n");

                last_word=0;
                logp=0;
                wordcn=0;
                while (1) {
                    word=readWordIndex(fi);     //read next word
                    computeNet(last_word, word);      //compute probability distribution
                    if (feof(fi)) break;        //end of file: report LOGP, PPL

                    if (word!=-1) {
                        logp+=log(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac);
                        wordcn++;
                    }

                    /*if (word!=-1)
                      fprintf(flog, "%d\t%f\t%s\n", word, neu2[word].ac, vocab[word].word);
                      else
                      fprintf(flog, "-1\t0\t\tOOV\n");*/

                    //learnNet(last_word, word);    //*** this will be in implemented for dynamic models
                    copyHiddenLayerToInput();

                    if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation

                    last_word=word;

                    for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
                    history[0]=last_word;

                    if (independent && (word==0)) netReset();
                }
                fclose(fi);

                fprintf(flog, "\niter: %d\n", iter);
                fprintf(flog, "valid log probability: %f\n", logp);
                fprintf(flog, "PPL net: %f\n", exp(-logp/(real)wordcn));

                fclose(flog);

                printf("VALID entropy: %.4f\n", -logp/wordcn);

                counter=0;
                train_cur_pos=0;

                if (logp<llogp)
                    restoreWeights();
                else
                    saveWeights();

                if (logp*min_improvement<llogp) {
                    if (alpha_divide==0) alpha_divide=1;
                    else {
                        saveNet();
                        break;
                    }
                }

                if (alpha_divide) alpha/=2;

                llogp=logp;
                logp=0;
                iter++;
                saveNet();
            }
        }



        void writeSynapse(const int M, const int N, const synapse * syn, const char * fname){
            int len = M *N ;
            vector<real> buffer(len);
            for (int b=0; b<M; b++) {
                for (int a=0; a<N; a++) {
                    int idx = a+b*N;
                    real w = syn[idx].weight;
                    buffer[idx] = w;
                }  
            }
            ofstream fl(fname, ios::out | ios::binary);
            fl.write(reinterpret_cast<char*>(&buffer[0]), sizeof(real)*len);
            fl.close();
            fl.clear();
        }

        void dumpSynapses(){
            writeSynapse(this->layer1_size, this->layer0_size, this->syn0, "/tmp/syn0init.bin");
            writeSynapse(this->layer2_size, this->layer1_size, this->syn1, "/tmp/syn1init.bin");
        }

        void dumpVocab(){
            ofstream fl("/tmp/voc.txt", ios::out);

            for(int it = 0; it < this->vocab_size; it++){
                //printf("%s\n", this->vocab[it].word);
                fl << this->vocab[it].word << " " << this->searchVocab(this->vocab[it].word) << "\n";
            }
            fl.close();
            fl.clear();
        }

        void printNeuron(int l, int mode){
            //neuron 0
            if(!(l >= 0 && l < 3)) {
                printf("WARNING: invalid layer. Returning");
                return;
            }
            vector<int> sizes = {this->layer0_size, this->layer1_size, this->layer2_size};
            vector< neuron * > ns = {this->neu0, this->neu1, this->neu2};

            const int sz = min(sizes[l], 10);
            const neuron * layer = ns[l];

            printf("Neuron layer %d:\t", l);
            for(int it = 0; it <sz; ++it){
                const neuron & n = layer[it];
                printf("%.8f ", mode == 0 ? n.ac : n.er);
            }
            printf("\n");

        }

        void printSynapse(int l){
            if(!(l >= 0 && l < 3)) {
                printf("WARNING: invalid synapse. Returning");
                return;
            }

            vector<int> sizes1 = {this->layer0_size, this->layer1_size, this->layer0_size};
            vector<int> sizes2 = {this->layer1_size, this->layer2_size, this->layer1_size};
            vector<synapse *> syns = {this->syn0, this->syn1, this->bptt_syn0};

            const int sz1 = min(sizes1[l], 10);
            const int sz2 = min(sizes2[l], 10);
            const synapse * svec = syns[l];

            switch(l){
                case 1:{
                           printf("\nWo\n");
                           for(int it = 0; it < sz1; it++){
                               for(int jt = 0; jt < sz2-1; jt++){
                                   const synapse & syn = svec[it + jt * sz1];
                                   printf("%.8f ", syn.weight);
                               }
                               printf("\n");
                           }
                           printf("\n");
                       }break;
                default:{
                            printf("\n");
                            if(l == 2)
                                printf("g");
                            printf("Wi\n");
                            for(int it = 0; it < sz1; it++){
                                if(it == this->vocab_size) {
                                    if(l == 2)
                                        printf("g");
                                    printf("Wh:\n");
                                }
                                for(int jt = 0; jt < sz2; jt++){
                                    const synapse & syn = svec[it + jt * sz1];
                                    printf("%.8f ", syn.weight);
                                }
                                printf("\n");
                            }
                            printf("\n");
                        }break;
            }
            //syn0: layer0 x layer1
            //syn1: layer1 x layer2
            //bptt_syn0: layer0 x layer1
            //printf("\nSynapse %d: \n", l);
            //for(int it = 0; it < sz1; it++){
            //    for(int jt = 0; jt < sz2; jt++){
            //        const synapse & syn = svec[it + jt * sz1];
            //        printf("%.8f ", syn.weight);
            //    }
            //    printf("\n");
            //}
            //printf("End Synapse %d: \n", l);
        }
};

int main(int argc, char **argv)
{
    int i;

    int debug_mode=1;

    int fileformat=TEXT;

    int train_mode=0;
    int valid_data_set=0;
    int test_data_set=0;
    int rnnlm_file_set=0;

    int alpha_set=0, train_file_set=0;

    int class_size=100;
    int old_classes=0;
    float lambda=0.75;
    float gradient_cutoff=15;
    float dynamic=0;
    float starting_alpha=0.1;
    float regularization=0.0000001;
    float min_improvement=1.003;
    int hidden_size=30;
    int compression_size=0;
    long long direct=0;
    int direct_order=3;
    int bptt=0;
    int bptt_block=10;
    int gen=0;
    int independent=0;
    int use_lmprob=0;
    int rand_seed=1;
    int nbest=0;
    int one_iter=0;
    int maxIter=INT_MAX;
    int anti_k=0;

    char train_file[MAX_STRING];
    char valid_file[MAX_STRING];
    char test_file[MAX_STRING];
    char rnnlm_file[MAX_STRING];
    char lmprob_file[MAX_STRING];

    FILE *f;

    if (argc==1) {
        //printf("Help\n");

        printf("Recurrent neural network based language modeling toolkit v 0.3d\n\n");

        printf("Options:\n");

        //
        printf("Parameters for training phase:\n");

        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train rnnlm model\n");

        printf("\t-class <int>\n");
        printf("\t\tWill use specified amount of classes to decompose vocabulary; default is 100\n");

        printf("\t-old-classes\n");
        printf("\t\tThis will use old algorithm to compute classes, which results in slower models but can be a bit more precise\n");

        printf("\t-rnnlm <file>\n");
        printf("\t\tUse <file> to store rnnlm model\n");

        printf("\t-binary\n");
        printf("\t\tRnnlm model will be saved in binary format (default is plain text)\n");

        printf("\t-valid <file>\n");
        printf("\t\tUse <file> as validation data\n");

        printf("\t-alpha <float>\n");
        printf("\t\tSet starting learning rate; default is 0.1\n");

        printf("\t-beta <float>\n");
        printf("\t\tSet L2 regularization parameter; default is 1e-7\n");

        printf("\t-hidden <int>\n");
        printf("\t\tSet size of hidden layer; default is 30\n");

        printf("\t-compression <int>\n");
        printf("\t\tSet size of compression layer; default is 0 (not used)\n");

        printf("\t-direct <int>\n");
        printf("\t\tSets size of the hash for direct connections with n-gram features in millions; default is 0\n");

        printf("\t-direct-order <int>\n");
        printf("\t\tSets the n-gram order for direct connections (max %d); default is 3\n", MAX_NGRAM_ORDER);

        printf("\t-bptt <int>\n");
        printf("\t\tSet amount of steps to propagate error back in time; default is 0 (equal to simple RNN)\n");

        printf("\t-bptt-block <int>\n");
        printf("\t\tSpecifies amount of time steps after which the error is backpropagated through time in block mode (default 10, update at each time step = 1)\n");

        printf("\t-one-iter\n");
        printf("\t\tWill cause training to perform exactly one iteration over training data (useful for adapting final models on different data etc.)\n");

        printf("\t-max-iter\n");
        printf("\t\tWill cause training to perform exactly <max-iter> iterations over training data (useful to test static learning rates if min-improvement is set to 0.0)\n");

        printf("\t-anti-kasparek <int>\n");
        printf("\t\tModel will be saved during training after processing specified amount of words\n");

        printf("\t-min-improvement <float>\n");
        printf("\t\tSet minimal relative entropy improvement for training convergence; default is 1.003\n");

        printf("\t-gradient-cutoff <float>\n");
        printf("\t\tSet maximal absolute gradient value (to improve training stability, use lower values; default is 15, to turn off use 0)\n");

        //

        printf("Parameters for testing phase:\n");

        printf("\t-rnnlm <file>\n");
        printf("\t\tRead rnnlm model from <file>\n");

        printf("\t-test <file>\n");
        printf("\t\tUse <file> as test data to report perplexity\n");

        printf("\t-lm-prob\n");
        printf("\t\tUse other LM probabilities for linear interpolation with rnnlm model; see examples at the rnnlm webpage\n");

        printf("\t-lambda <float>\n");
        printf("\t\tSet parameter for linear interpolation of rnnlm and other lm; default weight of rnnlm is 0.75\n");

        printf("\t-dynamic <float>\n");
        printf("\t\tSet learning rate for dynamic model updates during testing phase; default is 0 (static model)\n");

        //

        printf("Additional parameters:\n");

        printf("\t-gen <int>\n");
        printf("\t\tGenerate specified amount of words given distribution from current model\n");

        printf("\t-independent\n");
        printf("\t\tWill erase history at end of each sentence (if used for training, this switch should be used also for testing & rescoring)\n");

        printf("\nExamples:\n");
        printf("rnnlm -train train -rnnlm model -valid valid -hidden 50\n");
        printf("rnnlm -rnnlm model -test test\n");
        printf("\n");

        return 0;	//***
    }


    //set debug mode
    i=argPos((char *)"-debug", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: debug mode not specified!\n");
            return 0;
        }

        debug_mode=atoi(argv[i+1]);

        if (debug_mode>0)
            printf("debug mode: %d\n", debug_mode);
    }


    //search for train file
    i=argPos((char *)"-train", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: training data file not specified!\n");
            return 0;
        }

        strcpy(train_file, argv[i+1]);

        if (debug_mode>0)
            printf("train file: %s\n", train_file);

        f=fopen(train_file, "rb");
        if (f==NULL) {
            printf("ERROR: training data file not found!\n");
            return 0;
        }

        train_mode=1;

        train_file_set=1;
    }


    //set one-iter
    i=argPos((char *)"-one-iter", argc, argv);
    if (i>0) {
        one_iter=1;

        if (debug_mode>0)
            printf("Training for one iteration\n");
    }

    //set max-iter
    i=argPos((char *)"-max-iter", argc, argv);
    if (i>0) {

        if (i+1==argc) {
            printf("ERROR: maximum number of iterations not specified!\n");
            return 0;
        }

        maxIter=atoi(argv[i+1]);

        if (debug_mode>0)
            printf("Maximum number of iterations: %d\n", maxIter);
    }


    //search for validation file
    i=argPos((char *)"-valid", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: validation data file not specified!\n");
            return 0;
        }

        strcpy(valid_file, argv[i+1]);

        if (debug_mode>0)
            printf("valid file: %s\n", valid_file);

        f=fopen(valid_file, "rb");
        if (f==NULL) {
            printf("ERROR: validation data file not found!\n");
            return 0;
        }

        valid_data_set=1;
    }

    if (train_mode && !valid_data_set) {
        if (one_iter==0) {
            printf("ERROR: validation data file must be specified for training!\n");
            return 0;
        }
    }


    //set nbest rescoring mode
    i=argPos((char *)"-nbest", argc, argv);
    if (i>0) {
        nbest=1;
        if (debug_mode>0)
            printf("Processing test data as list of nbests\n");
    }


    //search for test file
    i=argPos((char *)"-test", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: test data file not specified!\n");
            return 0;
        }

        strcpy(test_file, argv[i+1]);

        if (debug_mode>0)
            printf("test file: %s\n", test_file);


        if (nbest && (!strcmp(test_file, "-"))) ; else {
            f=fopen(test_file, "rb");
            if (f==NULL) {
                printf("ERROR: test data file not found!\n");
                return 0;
            }
        }

        test_data_set=1;
    }


    //set class size parameter
    i=argPos((char *)"-class", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: amount of classes not specified!\n");
            return 0;
        }

        class_size=atoi(argv[i+1]);

        if (debug_mode>0)
            printf("class size: %d\n", class_size);
    }


    //set old class
    i=argPos((char *)"-old-classes", argc, argv);
    if (i>0) {
        old_classes=1;

        if (debug_mode>0)
            printf("Old algorithm for computing classes will be used\n");
    }


    //set lambda
    i=argPos((char *)"-lambda", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: lambda not specified!\n");
            return 0;
        }

        lambda=atof(argv[i+1]);

        if (debug_mode>0)
            printf("Lambda (interpolation coefficient between rnnlm and other lm): %f\n", lambda);
    }


    //set gradient cutoff
    i=argPos((char *)"-gradient-cutoff", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: gradient cutoff not specified!\n");
            return 0;
        }

        gradient_cutoff=atof(argv[i+1]);

        if (debug_mode>0)
            printf("Gradient cutoff: %f\n", gradient_cutoff);
    }


    //set dynamic
    i=argPos((char *)"-dynamic", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: dynamic learning rate not specified!\n");
            return 0;
        }

        dynamic=atof(argv[i+1]);

        if (debug_mode>0)
            printf("Dynamic learning rate: %f\n", dynamic);
    }


    //set gen
    i=argPos((char *)"-gen", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: gen parameter not specified!\n");
            return 0;
        }

        gen=atoi(argv[i+1]);

        if (debug_mode>0)
            printf("Generating # words: %d\n", gen);
    }


    //set independent
    i=argPos((char *)"-independent", argc, argv);
    if (i>0) {
        independent=1;

        if (debug_mode>0)
            printf("Sentences will be processed independently...\n");
    }


    //set learning rate
    i=argPos((char *)"-alpha", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: alpha not specified!\n");
            return 0;
        }

        starting_alpha=atof(argv[i+1]);

        if (debug_mode>0)
            printf("Starting learning rate: %f\n", starting_alpha);
        alpha_set=1;
    }


    //set regularization
    i=argPos((char *)"-beta", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: beta not specified!\n");
            return 0;
        }

        regularization=atof(argv[i+1]);

        if (debug_mode>0)
            printf("Regularization: %f\n", regularization);
    }


    //set min improvement
    i=argPos((char *)"-min-improvement", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: minimal improvement value not specified!\n");
            return 0;
        }

        min_improvement=atof(argv[i+1]);

        if (debug_mode>0)
            printf("Min improvement: %f\n", min_improvement);
    }


    //set anti kasparek
    i=argPos((char *)"-anti-kasparek", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: anti-kasparek parameter not set!\n");
            return 0;
        }

        anti_k=atoi(argv[i+1]);

        if ((anti_k!=0) && (anti_k<10000)) anti_k=10000;

        if (debug_mode>0)
            printf("Model will be saved after each # words: %d\n", anti_k);
    }


    //set hidden layer size
    i=argPos((char *)"-hidden", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: hidden layer size not specified!\n");
            return 0;
        }

        hidden_size=atoi(argv[i+1]);

        if (debug_mode>0)
            printf("Hidden layer size: %d\n", hidden_size);
    }


    //set compression layer size
    i=argPos((char *)"-compression", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: compression layer size not specified!\n");
            return 0;
        }

        compression_size=atoi(argv[i+1]);

        if (debug_mode>0)
            printf("Compression layer size: %d\n", compression_size);
    }


    //set direct connections
    i=argPos((char *)"-direct", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: direct connections not specified!\n");
            return 0;
        }

        direct=atoi(argv[i+1]);

        direct*=1000000;
        if (direct<0) direct=0;

        if (debug_mode>0)
            printf("Direct connections: %dM\n", (int)(direct/1000000));
    }


    //set order of direct connections
    i=argPos((char *)"-direct-order", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: direct order not specified!\n");
            return 0;
        }

        direct_order=atoi(argv[i+1]);
        if (direct_order>MAX_NGRAM_ORDER) direct_order=MAX_NGRAM_ORDER;

        if (debug_mode>0)
            printf("Order of direct connections: %d\n", direct_order);
    }


    //set bptt
    i=argPos((char *)"-bptt", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: bptt value not specified!\n");
            return 0;
        }

        bptt=atoi(argv[i+1]);
        bptt++;
        if (bptt<1) bptt=1;

        if (debug_mode>0)
            printf("BPTT: %d\n", bptt-1);
    }


    //set bptt block
    i=argPos((char *)"-bptt-block", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: bptt block value not specified!\n");
            return 0;
        }

        bptt_block=atoi(argv[i+1]);
        if (bptt_block<1) bptt_block=1;

        if (debug_mode>0)
            printf("BPTT block: %d\n", bptt_block);
    }


    //set random seed
    i=argPos((char *)"-rand-seed", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: Random seed variable not specified!\n");
            return 0;
        }

        rand_seed=atoi(argv[i+1]);

        if (debug_mode>0)
            printf("Rand seed: %d\n", rand_seed);
    }


    //use other lm
    i=argPos((char *)"-lm-prob", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: other lm file not specified!\n");
            return 0;
        }

        strcpy(lmprob_file, argv[i+1]);

        if (debug_mode>0)
            printf("other lm probabilities specified in: %s\n", lmprob_file);

        f=fopen(lmprob_file, "rb");
        if (f==NULL) {
            printf("ERROR: other lm file not found!\n");
            return 0;
        }

        use_lmprob=1;
    }


    //search for binary option
    i=argPos((char *)"-binary", argc, argv);
    if (i>0) {
        if (debug_mode>0)
            printf("Model will be saved in binary format\n");

        fileformat=BINARY;
    }


    //search for rnnlm file
    i=argPos((char *)"-rnnlm", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: model file not specified!\n");
            return 0;
        }

        strcpy(rnnlm_file, argv[i+1]);

        if (debug_mode>0)
            printf("rnnlm file: %s\n", rnnlm_file);

        f=fopen(rnnlm_file, "rb");

        rnnlm_file_set=1;
    }
    if (train_mode && !rnnlm_file_set) {
        printf("ERROR: rnnlm file must be specified for training!\n");
        return 0;
    }
    if (test_data_set && !rnnlm_file_set) {
        printf("ERROR: rnnlm file must be specified for testing!\n");
        return 0;
    }
    if (!test_data_set && !train_mode && gen==0) {
        printf("ERROR: training or testing must be specified!\n");
        return 0;
    }
    if ((gen>0) && !rnnlm_file_set) {
        printf("ERROR: rnnlm file must be specified to generate words!\n");
        return 0;
    }


    srand(1);

    if (train_mode) {
        CRnnLMaccess model1;

        model1.setTrainFile(train_file);
        model1.setRnnLMFile(rnnlm_file);
        model1.setFileType(fileformat);

        model1.setOneIter(one_iter);
        model1.setMaxIter(maxIter);
        if (one_iter==0) model1.setValidFile(valid_file);

        model1.setClassSize(class_size);
        model1.setOldClasses(old_classes);
        model1.setLearningRate(starting_alpha);
        model1.setGradientCutoff(gradient_cutoff);
        model1.setRegularization(regularization);
        model1.setMinImprovement(min_improvement);
        model1.setHiddenLayerSize(hidden_size);
        model1.setCompressionLayerSize(compression_size);
        model1.setDirectSize(direct);
        model1.setDirectOrder(direct_order);
        model1.setBPTT(bptt);
        model1.setBPTTBlock(bptt_block);
        model1.setRandSeed(rand_seed);
        model1.setDebugMode(debug_mode);
        model1.setAntiKasparek(anti_k);
        model1.setIndependent(independent);

        model1.alpha_set=alpha_set;
        model1.train_file_set=train_file_set;


        model1.trainNet();
    }

    if (test_data_set && rnnlm_file_set) {
        CRnnLMaccess model1;

        model1.setLambda(lambda);
        model1.setRegularization(regularization);
        model1.setDynamic(dynamic);
        model1.setTestFile(test_file);
        model1.setRnnLMFile(rnnlm_file);
        model1.setRandSeed(rand_seed);
        model1.useLMProb(use_lmprob);
        if (use_lmprob) model1.setLMProbFile(lmprob_file);
        model1.setDebugMode(debug_mode);

        model1.restoreNet();
        model1.dumpVocab();
        model1.dumpSynapses();
        if (nbest==0) model1.testNet();
        else model1.testNbest();
    }

    if (gen>0) {
        CRnnLMaccess model1;

        model1.setRnnLMFile(rnnlm_file);
        model1.setDebugMode(debug_mode);
        model1.setRandSeed(rand_seed);
        model1.setGen(gen);

        model1.testGen();
    }


    return 0;
}
