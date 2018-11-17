import numpy as np
import math

data = np.load('fed.npz')['arr_0']
authors = data[:,0]
vars = data[:,1:4]

H = np.array([2,3,-4])
M = np.array([1,5,1])
J = np.array([4,-2,-1])

def logit(freqs):
    h = math.exp(H.dot(freqs))
    m = math.exp(M.dot(freqs))
    j = math.exp(J.dot(freqs))
    tot = h+m+j
    return h/tot, m/tot, j/tot

probs = np.array([logit(v) for v in vars])

for a,(p1,p2,p3) in zip(authors,probs):
    print(a,p1,p2,p3)


