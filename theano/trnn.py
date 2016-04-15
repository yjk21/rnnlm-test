from numpy import *
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from scipy.special import expit
from scipy.misc import logsumexp
import time

def step(xt,htm1, Wh, Wi, nl = expit):
    at = dot(Wh, htm1) + Wi[:,xt] 
    ht = nl(at)
    return ht

def fwd(x,y, h0, Wh, Wi, Wo):
    ht = h0
    lp = 0.0
    lp = zeros(len(x))
    H = zeros((len(x), Wh.shape[1]))
    for t, (xt,yt) in enumerate(zip(x,y)):
        ht = step(xt, ht, Wh, Wi)
        zt = dot(Wo, ht)
        lpt = logSoftmax(zt, yt)
        lp[t] = lpt
        H[t,:] = ht
    return lp,H

def logSoftmax(v, idx):
    maxVal = v.max()
    return v[idx] - maxVal - log(sum(exp(v - maxVal)))

def tLogSoftmax(tvec, tidx):
    maxVal = T.max(tvec)
    return tvec[tidx] - maxVal - T.log(T.sum(T.exp(tvec - maxVal)))

def tStep(txt,thtm1, tWh, tWi, nl = sigmoid):
    tat = T.dot(tWh, thtm1) + tWi[:,txt] 
    tht = nl(tat)
    return tht

def tStepLoss(txt, tyt, thtm1, tWh, tWi, tWo, nl = sigmoid):
    tat = T.dot(tWh, thtm1) + tWi[:,txt] 
    tht = nl(tat)
    tzt = T.dot(tWo, tht)
    tlp = tLogSoftmax(tzt, tyt) #next item probability
    return tlp, tht

def tfwd(th0, tx, ty, tWh, tWi, tWo, nl = sigmoid):
    #tH, tupd = theano.scan(fn = tStep, outputs_info = th0, sequences = tx, non_sequences = [tWh, tWi])
    tH, tupd = theano.scan(fn = tStep, outputs_info = [None,th0], sequences = [tx,ty], non_sequences = [tWh, tWi, tWo])


#===== Symbolic variables

tWh = T.matrix()
tWi = T.matrix()
tWo = T.matrix()

tx = T.ivector()
ty = T.ivector()

th0 = T.vector()

tx1 = T.iscalar()

#===== Actual variables

random.seed(1234)

D = 55
M = 2000

Wh = random.randn(D,D)
Wi = random.randn(D,M)
Wo = random.randn(M,D)
h0 = random.randn(D)

seq = [0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0]
seq = random.randint(4, size=50000).astype(int32)
#x = array([1,2]).astype(int32)
#y = array([3,2]).astype(int32)
x = seq[:-1]
y = seq[1:]

tic = time.time()
lp0, H0 = fwd(x,y, h0, Wh, Wi, Wo) 
toc = time.time()
print "python impl:",toc - tic

def bla(tx,ty,thtm1, tWh, tWi, tWo):
    tat = T.dot(tWh, thtm1) + tWi[:, tx]
    tht = sigmoid(tat)
    tzt = T.dot(tWo, tht)
    tlpt = tLogSoftmax(tzt, ty)
    return tlpt, tht

tlp0 = T.scalar()

(a,b),tupd = theano.scan(fn = bla, sequences = [tx,ty],outputs_info = [None,th0], non_sequences = [tWh, tWi, tWo])
#c = a.eval({tx:x, ty:y, th0:h0, tWh :Wh, tWi:Wi, tWo: Wo})
tf = theano.function(inputs = [tx, ty, th0, tWh, tWi, tWo], outputs=[a,b[-1,:]], profile=False)

tic = time.time()
lp1, H1 = tf(x, y, h0, Wh, Wi, Wo)
toc = time.time()
print "theano loss:" ,toc - tic



#tH, _ = theano.scan(fn = tStep, outputs_info = th0, sequences = tx, non_sequences = [tWh, tWi])
#tZ = T.dot(tWo, tH.T)
#tN = T.log(T.sum(T.exp(tZ), axis=0))
#
#tfZ = theano.function(inputs = [tx, th0, tWh, tWi, tWo], outputs=tN)
#
#tic = time.time()
#Z = tfZ(x,  h0, Wh, Wi, Wo)
#toc = time.time()
#
#print "theano hidden:" ,toc - tic
