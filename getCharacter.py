import numpy as np
import h5py
import matplotlib.pyplot as plt

def sigmoid(X):
	den = 1.0 + np.exp(-X)
	return 1.0/den


def predict(theta1, theta2, init_X):
	m, n = init_X.shape
	X= np.hstack(( np.ones((m, 1)), init_X ))


	z2 = X.dot(theta1.T)
	a2 = sigmoid(z2)
	a2= np.hstack(( np.ones((m, 1)), a2 ))
	z3 = a2.dot(theta2.T)
	a3 = sigmoid(z3)

	p = a3.argmax(axis = 1)
	return p

def getChar(ch):
	data = h5py.File('theta.h5','r')
	theta1 = data['theta1'][:]
	theta2 = data['theta2'][:]
	data.close()


	img = ch.flatten()
	img = img.reshape((1, -1))
	
	num = predict(theta1, theta2, img)

	if(num < 10):
		return chr(num + 48)
	if(num < 36):
		num += 55
		return chr(num)
	num += 61
	return chr(num)


