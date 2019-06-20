import os
import io
import pickle
import random
import xlrd 
from pyvi import ViTokenizer
import gensim
import numpy as np
from scipy.sparse import coo_matrix

def read_xls(xls_file = 'comment_star.xls'):
	xlrd.biffh.unicode = lambda s, e: s.decode(e, errors="ignore")
	xlrd.book.unicode = lambda s, e: s.decode(e, errors="ignore")
	workbook = xlrd.open_workbook(xls_file, encoding_override="cp1251")
	sheet = workbook.sheet_by_name("Sheet 1")
	comments = []
	stars = []
	for rows in range(1,sheet.nrows):    
		comments.append( sheet.cell_value(rows,0) )
		stars.append( sheet.cell_value(rows,1) )
	return comments, stars
	
def split(comments, stars, test_len = 100):
	X_test = []
	y_test = []
	comments_len = len(comments)
	for i in range(test_len):
		ran_num = random.randint(0, comments_len - 1)
		X_test.append(comments[ran_num])
		del comments[ran_num]
		y_test.append(stars[ran_num])
		del stars[ran_num]
	return X_test, y_test
	
def to_vector(X_train, X_test):
	def tokenize(X_train):
		bag_of_word = set()	
		X = []
		for comment in X_train:
			comment = ViTokenizer.tokenize(comment)
			comment = gensim.utils.simple_preprocess(comment)
			for word in comment:	
				bag_of_word.add(word)
			X.append(comment)
		return X, bag_of_word
	
	X_train, bag_of_word = tokenize(X_train)
	X_test, dummy = tokenize(X_test)
	
	index_dict = {}
	i = 0
	for word in bag_of_word:
		index_dict[word] = i
		i += 1
		
	def feature_extract(X):
		row_id = []
		col_id = []
		data_id = []
		for i in range (len(X)): # row ith
			word_dict = {}
			for word in X[i]:
				try:
					col_id.append(index_dict[word])
					row_id.append(i)
					data_id.append(1)
				except KeyError:
					bug = False
		new_X = coo_matrix((data_id, (row_id, col_id)), shape=(len(X), len(bag_of_word)))
		return new_X
	
	X_train = feature_extract(X_train)
	X_test = feature_extract(X_test)
	return X_train, X_test
		
def main():
	print ('Reading from xls file')
	comments, stars = read_xls()
	print ('Read successfully')
	print ('Features extracting')
	X_test, y_test = split(comments, stars, 100)
	#At this, X_train = comments, y_train = stars
	y_train = stars
	X_train, X_test = to_vector(comments, X_test)
	print ('Extracted successfully')
	print ('Dumping')
	with open (r'X_train.pickle', 'wb') as file:
		pickle.dump(X_train, file)
		
	with open (r'y_train.pickle', 'wb') as file:
		pickle.dump(y_train, file)
		
	with open (r'X_test.pickle', 'wb') as file:
		pickle.dump(X_test, file)
		
	with open (r'y_test.pickle', 'wb') as file:
		pickle.dump(y_test, file)
	print ('Dumped successfully')
	
if __name__ == '__main__':
	main()
	