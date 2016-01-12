from numpy import *
from collections import deque
from scipy.misc import logsumexp
from scipy.special import expit


""" 
utilities related to rnnlm
"""

def createDbgData(prefix = 'dbg'):
    #s = array([0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0]).astype(int32)
    s = array([0,1,2,1,2]).astype(int32)
    x = s[:-1]
    y = s[1:]

    fid = open(prefix+'.tr','w')
    for it in s:
        fid.write(str(it) + " " )
    fid.close()

    fid = open(prefix+'.te','w')
    for it in s:
        fid.write(str(it) + " " )
    fid.close()

def loadParams(H, M):
    Wtemp = fromfile("tmp/syn0init.bin", dtype=float64)
    print "syn0 length:", Wtemp.shape, H ,M
    Wtemp = reshape(Wtemp, (H,H+M))
    Whi = Wtemp[:,:M]
    Whh = Wtemp[:,-H:]
    print Whi.shape
    print Whh.shape
    
    Wtemp = fromfile("tmp/syn1init.bin", dtype=float64)
    Woh = reshape(Wtemp, ((M+1),H))

    return Whi, Whh, Woh

def readVocab():
    
    #word ids
    words = []
    indices = []
    with open('tmp/voc.txt') as f:
        for line in f:
            print line
            toks = line.split(' ')
            if toks[0] != '</s>':
                words.append(int(toks[0]))
                indices.append(int(toks[1]))

    f.close()

    return dict(zip(words, indices))


def getData(wordMap,prefix='dbg' ):
    #tr = fromfile(prefix+'.tr', sep=' ', dtype=int64)
    #te = fromfile(prefix+'.te', sep=' ', dtype=int64)

    tr = []
    te = []
    with open(prefix + '.tr') as f:
        for line in f:
            tr.append(0)
            a = fromstring(line, dtype=int64, sep=' ')
            print line,a  
            tr.extend([wordMap[w] for w in a])
    tr.append(0)
    print tr

    with open(prefix + '.te') as f:
        for line in f:
            te.append(0)
            a = fromstring(line, dtype=int64, sep=' ')
            te.extend(a)

    te.append(0)

    return array(tr), array(te)


class bptthist:
    """
    Manages the bptt history, i.e. the sliding window into the sequence and corresponding hidden vectors
    """

    def __init__(self, h0, T, B=1):
        self.Hhist = deque([h0])
        self.HhistBwd = deque([zeros_like(h0)])
        self.xhist = deque([-1]) #some work around to keep xhist and Hhist the same length
        self.B = B
        self.T = T
    def get(self,t):
        return self.xhist[t], self.HhistBwd[t]

    def append(self, xt, ht):
        """append observation and hidden state. prune window if size is exceeded"""

        self.Hhist.appendleft(ht)
        self.HhistBwd.appendleft(ht)
        self.xhist.appendleft(xt)

        l = len(self.Hhist)
        #S = 3: Hhist should contain 4 elements
        if l - 1 > self.T:
            self.Hhist.pop()
            self.HhistBwd.pop()
            self.xhist.pop()

    #some tools for debugging
    def show(self):
        print  "BPTTHist"
        for it, (h, hb, x) in enumerate(zip(self.Hhist, self.HhistBwd, self.xhist)):
            print it, " h: ", h," hb: ", hb, " x:", x

    def compare(self, other):
        print len(self.Hhist) - len(other.Hhist)
        print len(self.HhistBwd) - len(other.HhistBwd)
        for h1, h2, h3, h4 in zip(self.Hhist, other.Hhist, self.HhistBwd, other.HhistBwd):
            print linalg.norm(h1-h2), linalg.norm(h3 - h4)


#step size strategies
class OptStep:
    def __init__(self, stepsize):
        self.stepsize = stepsize
    def set(self, new):
        self.stepsize = new
    def reset(self):
        pass
    def get(self, it, var,  grad):
        pass

class ConstStep(OptStep):
    def get(self, it, var, grad):
        return self.stepsize * grad

class RmsProp(OptStep):
    def __init__(self, var, stepsize = 1.0, decay = 0.9):
        OptStep.__init__(self,stepsize) 
        self.decay = decay
        self.cache = zeros_like(var)

    def get(self, it, var,  grad):
        self.cache = self.decay * self.cache + (1-self.decay) * grad**2
        return self.stepsize * grad / sqrt(self.cache + 1e-8)

    def reset(self):
        self.cache.fill(0.0)

class Adam(OptStep):
    def __init__(self, var, stepsize=0.001, beta1=0.9, beta2 =0.99, eps = 1e-8):
        OptStep.__init__(self, stepsize)
        self.beta1 = beta1
        self.beta2 = beta2
        self.b1t = 1.0
        self.b2t = 1.0
        self.eps = eps
        self.m = zeros_like(var)
        self.v = zeros_like(var)

    def reset(self):
        self.m.fill(0.0)
        self.v.fill(0.0)
        self.b1t = 1.0
        self.b2t = 1.0

    def get(self, it, var,  grad):
        self.m = self.beta1 * self.m + (1-self.beta1) * grad
        self.v = self.beta2 * self.v + (1-self.beta2) * (grad**2)
        self.b1t *= self.beta1
        self.b2t *= self.beta2
        mc = self.m / (1.0-self.b1t)
        vc = self.v / (1.0-self.b2t)
        return self.stepsize * mc / (sqrt(vc) + self.eps)


class mrnn2:
    """implements forward and backward pass. the RNN is parameterized by weight matrices Wo (output), Wi (input), Wh (hidden)"""
    def __init__(self, Wo, Wi, Wh, h0):
        self.gWo = zeros_like(Wo)
        self.gWi = zeros_like(Wi)
        self.gWh = zeros_like(Wh)

    def stepFwd(self, xt, xnext, htm1, Wo, Wi, Wh):
        """compute hidden state for xt and output and loss of xnext"""
        at = dot(Wh, htm1) + Wi[:,xt]
        ht = expit(at)
        zt = dot(Wo, ht)

        #Model specific: Softmax Layer
        maxz = max(zt)
        lPt = zt - maxz - logsumexp(zt - maxz)
        fpt = ht * (1.0 - ht)
        lt = lPt[xnext]
        glz = -exp(lPt)
        glz[xnext] += 1
        gla = dot(glz.T, Wo) * fpt
        self.gWo[:] = outer(glz, ht)

        return ht, lPt, fpt, gla, glz, lt

    def bwd(self, glT, xt, htm1, xhist, Hhist, glthist, Wo, Wi, Wh, B):
        #the only difference between batch and online is the number of glt terms that are present
        #there is at least one of them at coming from the loss at the latest timestep
        #for online that is all, whereas batch has a backlog of glt terms that are added to d L/d at 
        self.gWi.fill(0.0)

        gAt = glT

        #TODO: self.gWh[:] = outer(gAt, Hhist[0])
        self.gWh[:] = outer(gAt, htm1)
        self.gWi[:, xt] = gAt
        print "gWh@0: gla: ", gAt
        print self.gWh

        for t in xrange(len(Hhist)-1):

            xt = xhist[t]
            ht = Hhist[t]
            fp = ht * (1.0 - ht)
            Jac = Wh * fp 
            gAt = dot(gAt, Jac) 

            if t < len(glthist):
                gAt += glthist[t]

            self.gWi[:, xt] += gAt
            self.gWh += outer(gAt, Hhist[t+1]) 

            print "gWh@",t+1, " gla: ", gAt
            print self.gWh
            #print "class gWh@", t+1
            #print "gWh:", gAt, Hhist[t+1]
            #print self.gWh.T

            if t >= B-3:
                break #some more shenanigans by our friend...

def processSequence(data, Wo0, Wi0, Wh0, h0, lambdaReg, B, R, T, sso, ssi, ssh, val = None):
    """
    update the rnnlm model on a single sequence
    TODO: right now the methods of mrnn2 are somewhat pure. The only side effect is the computation of the gradient in the g-member variables, which can be removed. This makes the use of a class/mrnn2 instance basically obsolete.
    """
    updates, grads = [], []
    
    hist = bptthist(h0, T+B, B)
    
    glthist = deque()

    Wh = copy(Wh0)
    Wi = copy(Wi0)
    Wo = copy(Wo0)

    rnn = mrnn2(Wo, Wi, Wh, h0)

    for it, (xt, xnext) in enumerate(zip(data[:-1], data[1:])):

        c = it + 1
        if c % 500 == 0:
            print c
            if not val is None:
                print rnnLogProb(val, Wo, Wi, Wh, h0)[0]


        htm1 = hist.Hhist[0] 

        ht, lPt, fpt, gla, glz, lt = rnn.stepFwd(xt, xnext, htm1, Wo, Wi, Wh)

        print "=============================================="
        print "== START PARAMETER UPDATE Iteration", c
        print "=============================================="
        print "htm1"
        print htm1
        print "ht"
        print ht
        print "Wi:"
        print Wi.T, "\n"
        print "Wh:"
        print Wh.T, "\n"
        print "Wo:"
        print Wo.T, "\n"
        print "glt:", gla, "gamma:", gla * ht*(1.0-ht)
        print "xt:", xt, "xnext:", xnext, "ht:",ht
#        print "glz:", glz
        hist.show()

        deltao = sso.get(c, Wo, rnn.gWo)
        
        if c % R == 0:
            Wo += deltao - lambdaReg * Wo
        else:
            Wo += deltao 
        
        if c % B != 0: 
            glthist.appendleft(gla)
        else:
            #BPTT backward pass
            rnn.bwd(gla, xt, htm1, hist.xhist, hist.HhistBwd, glthist, Wo, Wi, Wh, B+T)
            print "gWi:"
            print rnn.gWi.T
            print "gWh:"
            print rnn.gWh.T

            deltah = ssh.get(c, Wh, rnn.gWh)
            deltai = ssi.get(c, Wi, rnn.gWi)

            if c % R == 0:
                Wh += deltah - lambdaReg * Wh
                Wi += deltai - lambdaReg * Wi
            else:
                Wh += deltah  
                Wi += deltai  

            glthist.clear()

        hist.append(xt, ht)

    return Wo, Wi, Wh

def rnnLogProb(data, Wo, Wi, Wh, h0):
    """Compute the log probability of a sequence under the RNN model specified by the given parameters"""
    htm1 = h0

    rnn = mrnn2(Wo, Wi, Wh, h0)
    lpall = zeros(len(data)-1)

    for it, (xt, xnext) in enumerate(zip(data[:-1], data[1:])):
        ht, lPt, fpt, gla, glz, lt = rnn.stepFwd(xt, xnext, htm1, Wo, Wi, Wh)
        htm1 = ht
        lpall[it] = lt
    return mean(lpall),lpall
