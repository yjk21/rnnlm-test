#!/bin/bash

#This is simple example how to use rnnlm tool for training and testing rnn-based language models
#Check 'example.output' how the output should look like
#SRILM toolkit must be installed for combination with ngram model to work properly


#rnn model is trained here
time ./rnnlm -test valid -rnnlm model 
