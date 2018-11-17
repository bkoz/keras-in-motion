import numpy as np
import matplotlib.pyplot as plt
plt.ion()

data = np.load('more_food.npz')['arr_0']
variables = data[:,:2]
cost = data[:,2]

plt.plot(variables[:,0],'bo')
plt.plot(variables[:,1],'g+')

plt.scatter(variables[:,0], variables[:,1],
        s=cost*100)

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(1, input_shape=(2,)))
model.add(Activation('linear'))

sgd = SGD(0.01)

model.compile(loss='mse',
              optimizer=sgd,
              metrics=['mse'])

H = model.fit(variables, cost, nb_epoch = 100)

cost_p = model.predict(variables)

plt.plot(variables[:,0], cost, 'bo',
        variables[:,0], cost_p,'g+')

plt.plot(variables[:,1], cost, 'bo',
        variables[:,1], cost_p,'g+')

W = model.get_weights()
print(W)


plt.plot(H.history['mean_squared_error'])










