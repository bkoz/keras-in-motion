import matplotlib.pyplot as plt
import numpy as np

plt.ion()

#read in some data
dfile = 'food.npz'
data = np.load(dfile)['arr_0']
grams = data[:,0]
cost = data[:,1]

plt.plot(grams,cost,'o')
plt.xlabel('Mass (g)')
plt.ylabel('Cost ($)')


n = 100
true_weights = np.array([[2,.5]])

X = np.array([np.random.random(n), np.ones(n)]).T
errs = (np.random.random(n)-.5)/5
Y_meas = true_weights.dot(X.T)[0,:] + errs

weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y_meas)

print(true_weights,weights)

test_pts = np.array([[0,1],[1,1]])
test_pred = weights.dot(test_pts.T)
plt.plot(X[:,0],Y_meas, 'bo',
    test_pts[:,0],test_pred,'g-')



