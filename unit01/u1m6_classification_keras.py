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

# get keras things, ok to copy from previous code
import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# specify model
model = Sequential()
model.add(Dense(3, input_shape=(3,))) # 3 authors = 3 outputs
model.add(Activation('softmax')) # no longer just linear
sgd = SGD()

model.compile(loss='categorical_crossentropy', optimizer=sgd,
             metrics = ['accuracy'])
# categorical crossentropy is a fancy way of measuring how close
# the correct author probability is to 1.0.  So we get credit for being closer
# accuracy is if the author with the highest probability is the 
# actual author, even if the results are close

# get some better variables
vars = data[:,[12,72,58]]

# get undisputed papers
known = authors != 4.
disp = authors == 4.

# Need to put authors into "1-hot" format
from keras.utils import np_utils
auth_cat = np_utils.to_categorical(authors[known])

# train
H = model.fit(vars[known], auth_cat,
              nb_epoch = 10)

probs = model.predict_proba(vars[disp])
print(authors[disp], probs)

