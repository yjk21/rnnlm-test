from numpy import *
import time
import sys
import os

from os.path import expanduser
home = expanduser("~")

import rnnutils as ru
import cPickle as pickle

reload(ru)

if __name__ == "__main__":

    os.system("make")

    #we run rnnlm in order to generate initial values and vocabulary
    os.system("./rnnlmMod -maxIter 1 -train dbglfm0.tr -valid dbglfm0.tr -rnnlm model -hidden 3 -rand-seed 1 -debug 20 -class 1 -bptt 4 -bptt-block 10 -direct-order 0 -direct 0 -binary")

    set_printoptions(precision=8)
    random.seed(666)

    alpha = 0.1
    beta = 1e-7

    beta2 = alpha * beta

    D = 3  #hidden dimensions
    R = 10 #regularization intervals
    B = 3  #Batch size
    T = 3  #BPTT window size 

    wordMap = ru.readVocab()
    M = len(wordMap)+1

    xtr, xte = ru.getData(wordMap,prefix='dbglfm0')
    Wi0, Wh0, Wo0 = ru.loadParams(D,M)
    Wo0 = Wo0[:M,:]

    print Wo0.shape

    sys.exit()

    h0 = (0.1*ones(D))

    #step size policy: constant step size
    alphaConst = alpha #learning rate
    sso = ru.ConstStep(alphaConst)
    ssh = ru.ConstStep(alphaConst)
    ssi = ru.ConstStep(alphaConst)

    #regularization strength
    lambdaReg = alpha * beta

    e,f,g = ru.processSequence(xtr, Wo0, Wi0, Wh0, h0, lambdaReg, B, R, T, sso, ssi, ssh)

    print ru.rnnLogProb(xtr, e,f,g, h0)[0]
