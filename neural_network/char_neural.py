import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import OneHotEncoder
import h5py


def sigmoid(X):
	den = 1.0 + np.exp(-X)
	return 1.0/den

def sigmoidGrad(X):
	y = sigmoid(X)
	return y * (1 - y)


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

def cost(params, init_X, y, Lamda, input_size,  num_labels, hidden_size):
	m, n = init_X.shape
	X= np.hstack(( np.ones((m, 1)), init_X ))

	theta1 = np.array(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
	theta2 = np.array(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))

	z2 = X.dot(theta1.T)
	a2 = sigmoid(z2)
	a2= np.hstack(( np.ones((m, 1)), a2 ))
	z3 = a2.dot(theta2.T)
	hx = sigmoid(z3)

	#J = -y * np.log(hx) -(1 - y) * np.log(1 - hx)

	J = -y * np.log(hx) -(1 - y) * np.log(1 - hx)
	J = J.sum() / m
	theta2[:,0] = 0
	theta1[:, 0] = 0
	
	theta1sq = theta1 ** 2
	theta2sq = theta2 ** 2

	sum = theta1sq.sum() + theta2sq.sum();
	J = J + sum * Lamda * 1.0 /(2 * m)

	
	#now calculating the gradient



	d3 = hx - y #d3-> m X 10

	#we dont need the 1st column of the theta2 to calcudate d2
	theta2cpy = theta2[:, 1:] 
	d2 = (theta2cpy.T.dot(d3.T)) * sigmoidGrad(z2).T;

	delta1 = d2.dot(X)
	delta2 = d3.T.dot(a2)

	delta1 = delta1 / m * 1.0
	delta2 = delta2 / m * 1.0

	Theta1_grad = delta1 + theta1 * Lamda / m * 1.0
	Theta2_grad = delta2 + theta2 * Lamda / m * 1.0

	grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))

	return J, grad

#loads the data from the file
def getData():
	data = h5py.File('dataset_9.h5','r')
	dataset = data['dataset'][:]
	data.close()

	np.random.shuffle(dataset)

	m, n = dataset.shape
	n = n-1

	#splitting the data into different sets
	train, validate, test = np.split(dataset, [int(.6*m), int(.8*m)])

	#converting into float
	X_train = train[:,0:n] * 1.0
	#feature scaling
	X_train = X_train / 255
	y_train = train[:, [n]]

	#converting into float
	X_test = test[:,0:n] * 1.0
	#feature scaling
	X_test = X_test / 255
	y_test = test[:, [n]]

	#converting into float
	X_validate = validate[:,0:n] * 1.0
	#feature scaling
	X_validate = X_validate / 255
	y_validate = validate[:, [n]]


	encoder = OneHotEncoder(sparse = False)

	y_onehot = encoder.fit_transform(y_train)

	return (np.hstack((X_train, y_onehot)), X_validate, y_validate, X_test, y_test)

def run():

	#loading data
	#np.set_printoptions(threshold=np.nan)
	training_data, X_val, y_val, X_test, y_test = getData()
	m, n = X_val.shape# n is 28*28
	

	input_size = n
	hidden_size = 100
	num_labels = 62	#26+26+10
	
	batch_size = 10000
	ran = 10 #no of times repeat
	

	Lamda = input("Enter lambda : ")
	num_iter = input("Enter num_iter : ")

	#randomly initializing the initial parameteres
	params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
	theta =  params
	#now training begins..
	for i in range(ran):

		print "Training %dth batch initial cost : " %i,

		np.random.shuffle(training_data)#shuffling the data

		X_train = training_data[:, :n]
		y_train = training_data[:, n:]


		X_batch = X_train[:batch_size ,:]
		y_batch = y_train[:batch_size ,:]

		J, grad = cost(theta, X_val, y_val, Lamda, input_size, num_labels, hidden_size)
		print J

		y_batch_val = y_batch.argmax(axis = 1)
		y_batch_val = y_batch_val.reshape(len(y_batch_val), 1)
	
		

		theta1 = np.array(theta[: hidden_size * (input_size + 1)].reshape(hidden_size, (input_size + 1)))
		theta2 = np.array(theta[hidden_size * (input_size + 1) : ].reshape(num_labels, (hidden_size + 1)))



		pred = predict(theta1, theta2, X_batch)
		pred = pred.reshape(len(pred), 1)
		corr = (pred == y_batch_val) * 1;

		acc =  1.0 *sum(corr)/len(corr)
		print "Accuracy in batch is : %f" %(acc*100)

		
		pred = predict(theta1, theta2, X_val)
		pred = pred.reshape(len(pred), 1)
		corr = (pred == y_val) * 1;

		acc =  1.0 *sum(corr)/len(corr)
		print "Accuracy in validation set is : %f" %(acc*100)



		pred = predict(theta1, theta2, X_test)
		pred = pred.reshape(len(pred), 1)
		corr = (pred == y_test) * 1;

		acc =  1.0 *sum(corr)/len(corr)
		print "Accuracy in test is : %f" %(acc*100)

		#preforming minimization of cost function

		Result = opt.minimize(fun = cost, x0 = theta, args = (X_batch, y_batch, Lamda, input_size, num_labels, hidden_size),
		 		method = 'TNC', jac = True, options = {'maxiter': num_iter})
		theta = Result.x

		theta1 = np.array(theta[: hidden_size * (input_size + 1)].reshape(hidden_size, (input_size + 1)))
		theta2 = np.array(theta[hidden_size * (input_size + 1) : ].reshape(num_labels, (hidden_size + 1)))

		pred = predict(theta1, theta2, X_batch)
		pred = pred.reshape(len(pred), 1)
		corr = (pred == y_batch_val) * 1;

		acc =  1.0 *sum(corr)/len(corr)
		print "Accuracy in batch after descent is : %f \n" %(acc*100)
		



	print "Training done...	"

	#gettind the parametes from rheta
	theta1 = np.array(theta[: hidden_size * (input_size + 1)].reshape(hidden_size, (input_size + 1)))
	theta2 = np.array(theta[hidden_size * (input_size + 1) : ].reshape(num_labels, (hidden_size + 1)))





	
	pred = predict(theta1, theta2, X_val)
	pred = pred.reshape(len(pred), 1)
	corr = (pred == y_val) * 1;

	acc =  1.0 *sum(corr)/len(corr)
	print "Accuracy in validation set is : %f" %(acc*100)



	pred = predict(theta1, theta2, X_test)
	pred = pred.reshape(len(pred), 1)
	corr = (pred == y_test) * 1;

	acc =  1.0 *sum(corr)/len(corr)
	print "Accuracy in test is : %f" %(acc*100)


	#saving the parameters
	h5f = h5py.File('theta.h5', 'w')

	h5f.create_dataset('theta1', data=theta1)
	h5f.create_dataset('theta2', data=theta2)
	h5f.close()
	
	

if __name__ == "__main__":
	run();
