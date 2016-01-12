#!/bin/bash

#This is simple example how to use rnnlm tool for training and testing rnn-based language models
#Check 'example.output' how the output should look like
#SRILM toolkit must be installed for combination with ngram model to work properly

mkdir -p tmp
make clean
make

rm -f model
rm -f model.output.txt

#rnn model is trained here
time ./rnnlmMod -maxIter 1 -train dbglfm0.tr -valid dbglfm0.tr -rnnlm model -hidden 3 -rand-seed 1 -debug 20 -class 1 -bptt 4 -bptt-block 10 -direct-order 0 -direct 0 -binary
