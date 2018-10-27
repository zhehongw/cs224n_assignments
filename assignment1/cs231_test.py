#!/usr/bin/env python 
import numpy as np
import matplotlib.pyplot as plt
from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive
from q2_neural import forward_backward_prop

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
labels = np.zeros((N*K, K))
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
  labels[ix, j] = 1
# lets visualize the data:
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()
dimensions = [D, 100, K]
params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
    dimensions[1] + 1) * dimensions[2], )

step_size = 1e-0
for i in xrange(10000):
    cost, grad = forward_backward_prop(X, labels, params, dimensions)
    if i % 1000 == 0:
        print "iteration %d: loss %f" % (i, cost)
    params += -step_size * grad 

ofs = 0
Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
ofs += Dx * H
b1 = np.reshape(params[ofs:ofs + H], (1, H))
ofs += H
W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
ofs += H * Dy
b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

z1 = X.dot(W1)+b1
a1 = sigmoid(z1)
z2 = a1.dot(W2)+b2
output = softmax(z2)
predicted_class = np.argmax(output, axis=1)

print 'training accuracy: %.2f' % (np.mean(predicted_class == y))
