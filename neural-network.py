import numpy as np 

# sigmoid function
def nonlinear(x, derivate = False):
    if(derivate == True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

# input dataset
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]])

# output dataset
Y = np.array([
    [0, 0, 1, 1]]).T

# create random to make calculus
np.random.seed(1)

# initiaize weights randomly with mean 0
syn0 = 2*np.random.random((3, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

for iteration in range(10000):

    l0 = X
    l1 = nonlinear(np.dot(l0, syn0))
    l2 = nonlinear(np.dot(l1, syn1))

    # error l2
    l2_error = Y - l2

    # delta l2
    l2_delta = l2_error * nonlinear(l2, True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlinear(l1, derivate = True)
    
    # update the weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print("Output after training: ")
print(l1)