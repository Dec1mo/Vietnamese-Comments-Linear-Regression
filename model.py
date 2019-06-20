import pickle
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix

def calc_w(X_train, y_train):
	'''
	XT = X_train.transpose()
	multi_X = XT.multiply(X_train)
	inv_multi_X = inv(multi_X)
	mul_X = inv_multi_X.multiply(XT)
	y = csr_matrix(y_train)
	inv_y = inv(y)
	return mul_X.multiply(inv_y)
	'''
	return (inv(X_train.transpose() @ X_train) @ (X_train.transpose())) @ (csr_matrix(y_train).transpose())
	
def predict(w, X_test):
	y_predict = (X_test @ w).todense()
	new_y = [one_y[0] for one_y in y_predict]
	return y_predict
	
def evaluate(y_predict, y_test):
	for i in range (len(y_test)):
		print("{}. predict = {}\t test = {}".format(i+1, y_predict[i], y_test[i]));

def main():
	print ('Loading pickle files')
	with open (r'X_train.pickle', 'rb') as file:
		X_train = pickle.load(file)
	with open (r'X_test.pickle', 'rb') as file:
		X_test = pickle.load(file)
	with open (r'y_train.pickle', 'rb') as file:
		y_train = pickle.load(file)
	with open (r'y_test.pickle', 'rb') as file:
		y_test = pickle.load(file)
	print ('Loaded successfully')
	
	w = calc_w(X_train, y_train)
	
	with open (r'w.pickle', 'wb') as file:
		pickle.dump(w, file, pickle.HIGHEST_PROTOCOL)
	
	y_predict = predict(w, X_test)
	evaluate (y_predict, y_test)	
	
if __name__ == '__main__':
	main()