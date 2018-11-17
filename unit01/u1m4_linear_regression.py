import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# Load data with numpy
data = np.load('more_food.npz')['arr_0']
variables = data[:,:2] # First 2 cols
cost = data[:,2]

# plot against each variable
plt.plot(variables[:,0],cost,'o') # mass
plt.plot(variables[:,1],cost,'+') # temp

# Plot with size equal to cost
plt.scatter(variables[:,0],variables[:,1],s=cost*100)

import keras
from keras.models import Sequential # provides basic model object
# to which we add computations, single input/output
from keras.layers import Dense, Activation # every operation is a
# "layer".  A sequential model is just a bunch of ordered computations
from keras.optimizers import SGD # stochastic gradient descent, 
# this provides computations to train the model



# specify model
model = Sequential() # make a new empty model, a sequence of operations
model.add(Dense(1, input_shape=(2,))) # Dense just means give a
# weight to each variable + one bias variable.  '1' is number of
# output computed values.  You get a weight for each input for 
# every output, so the number of weights is (input+1)*output. 
# This grows quickly, so be careful
model.add(Activation('linear')) # How to transform data; 
# we're doing linear regression, so we want linear, 
# z = ax + by + c

# setup training
sgd = SGD(0.01) # create instance of the optimizer; 
# value is 'learning rate', or fraction of gradient to advance.
# Fast & reckless vs slow and careful

model.compile(loss='mse', optimizer = sgd, metrics=['mse']) 
# assign optimizer and define how we judge model, here Mean Square Error

# Do the training
H = model.fit(variables, cost, nb_epoch=100) # num_epochs is how
# many times to look at the data; because of gradient, you 
# likely need to train on a datapoint multiple times to extract
# full information...though you may "overtrain" which is like 
# memorizing the data rather than understanding it.


plt.plot(H.history['mean_squared_error']) # Remember when we made

# see what the model says about our data
cost_p = model.predict(variables)

# plot against actual data
plt.plot(variables[:,0],cost,'bo',
         variables[:,0],cost_p, 'g+') # mass
plt.plot(variables[:,1],cost,'bo',
         variables[:,1],cost_p, 'g+') # temp

# notice close matching with mass but not as clear with temp

# all in one plot, but not as clear what's going on
plt.scatter(variables[:,0],variables[:,1],c='b',s=100*cost)
plt.scatter(variables[:,0],variables[:,1],c='g',s=100*cost_p)

W = model.get_weights()
print(W)
