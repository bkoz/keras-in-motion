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

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(3, input_shape=(3,)))
model.add(Activation('softmax'))

sgd = SGD()
model.compile(loss="categorical_crossentropy",
            optimizer = sgd,
            metrics = ['accuracy'])

known = authors != 4.
disp = authors == 4.

vars = data[:,[12,72,58]]

from keras.utils import np_utils
auth_cat = np_utils.to_categorical(authors[known])
print(auth_cat[:2])

H = model.fit(vars[known], auth_cat, nb_epoch= 100)

probs = model.predict_proba(vars[disp])
print(probs)

















