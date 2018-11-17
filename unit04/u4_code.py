from keras.datasets import mnist
import numpy as np

(train_imgs, train_labs), (test_imgs, test_labs) = mnist.load_data()

# normalize
train_imgs = train_imgs.astype(np.float32) /255.
test_imgs = test_imgs.astype(np.float32) /255.

aet = 40000
aev = 40000+10000


nrows = 28
ncols = 28

import matplotlib.pyplot as plt
plt.ion()

plt.matshow(train_imgs[2], cmap= plt.cm.gray_r)

print(train_labs[2])

# modeling imports
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Reshape

# upsample is reverse of pooling, Deconv is reverse of conv
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D, ZeroPadding2D
import sklearn.metrics as metrics

model = Sequential()
model.add(Dense(128,input_shape=(nrows*ncols,)))
model.add(Activation('relu'))
model.add(Dense(nrows*ncols))
model.add(Activation('sigmoid'))

# Binary cross-entropy is possible because we are 0-1, else consider MSE
model.compile(loss='binary_crossentropy', optimizer='adadelta')

# train for a bit
h = model.fit(train_imgs.reshape([-1,784])[:aet],
          train_imgs.reshape([-1,784])[:aet],
          nb_epoch = 100,
          validation_data = (train_imgs.reshape([-1,784])[aet:aev],
                             train_imgs.reshape([-1,784])[aet:aev]))


plt.matshow(model.predict(train_imgs[2].reshape([1,784])).reshape([28,28]), cmap= plt.cm.gray)

# convolutional model
model = Sequential()
model.add(Convolution2D(24,5,5,input_shape=(1,nrows,ncols,),border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(24,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(24,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))

# decoding

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(216))
model.add(Activation('relu'))
model.add(Reshape((24,3,3)))

model.add(Convolution2D(24,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))
model.add(ZeroPadding2D(padding = (0,1,0,1)))

model.summary()


model.add(Convolution2D(24,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Convolution2D(24,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))

# reduce to 1 channel
model.add(Convolution2D(1,5,5, border_mode='same'))
model.add(Activation('sigmoid'))

# Binary cross-entropy is possible because we are 0-1, else consider MSE
model.compile(loss='binary_crossentropy', optimizer='adadelta')

# train for a bit
h = model.fit(train_imgs.reshape([-1,1,28,28])[:aet],
          train_imgs.reshape([-1,1,28,28])[:aet],
          nb_epoch = 40,
          validation_data = (train_imgs.reshape([-1,1,28,28])[aet:aev],
                             train_imgs.reshape([-1,1,28,28])[aet:aev]))


plt.matshow(model.predict(train_imgs[2].reshape([-1,1,28,28])).reshape([28,28]), cmap= plt.cm.gray)

# Transfer learning video

tlt = 50000

# Convert labels to one-hot for testing

from keras.utils.np_utils import to_categorical

train_cat_labs = to_categorical(train_labs)
test_cat_labs = to_categorical(test_labs)

print(train_cat_labs[2])

# Get relevant weights
W = model.get_weights()

# Recreate encoding model
model = Sequential()
model.add(Convolution2D(24,5,5,input_shape=(1,nrows,ncols,),border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(24,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(24,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))

# set the autoencoded weights, only from encoding layers
model.set_weights(W[:8])

# continue as normal

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd',
        metrics = ['accuracy'])

h = model.fit(train_imgs.reshape([-1,1,28,28])[tlt:],
          train_cat_labs[tlt:],
          nb_epoch = 10,
          validation_data = (test_imgs.reshape([-1,1,28,28]),
                             test_cat_labs))

model.save_weights('transfer_model.h5')

