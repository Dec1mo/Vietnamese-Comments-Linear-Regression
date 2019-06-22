import pickle
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import csr_matrix
import re
from sklearn import linear_model

def calc_w(X_train, y_train):
	inv_XXT = pinv((X_train.transpose() @ X_train).todense())
	M = (X_train.transpose() @ csr_matrix(y_train).transpose()).todense()
	return inv_XXT * M
	
def predict(w, X_test):
	y_predict = X_test.todense() * w
	new_y = [list(np.array(one_y[0]).reshape(-1,))[0] for one_y in y_predict]
	return new_y
	
def evaluate(y_predict, y_test, file_name = 'result.txt'):
	f = open(file_name,"a+")
	for i in range (len(y_test)):
		f.write("{}. predict = {:^20}\t test = {}\n".format(i+1, y_predict[i], y_test[i]));
		print("{}. predict = {:^20}\t test = {}".format(i+1, y_predict[i], y_test[i]));
	f.close() 
	
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
	print ("X_train's rows (number of train samples) = {}".format (len(X_train.todense())))
	print ("y_train's rows (number of train label-number samples) = {}".format (len(y_train)))
	print ("X_test's rows (number of test samples) = {}".format (len(X_test.todense())))
	print ("y_test's rows (number of test label-number samples) = {}".format (len(y_test)))
	
	'''
	w = calc_w(X_train, y_train)
	print ('Dumping w to pickle file successfully')
	with open (r'w.pickle', 'wb') as file:
		pickle.dump(w, file, pickle.HIGHEST_PROTOCOL)
	print ('Dumped w to pickle file successfully')
	'''
	with open (r'w.pickle', 'rb') as file:
		w = pickle.load(file)
	
	#y_predict = predict(w, X_test)
	#evaluate (y_predict, y_test)	
	
	# For libarary-based version
	regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
	regr.fit(X_train, y_train)
	print( 'Solution found by scikit-learn  : ', regr.coef_ )
	print( 'Solution found by scikit-learn  : ', w.T )
	
	
if __name__ == '__main__':
	main()