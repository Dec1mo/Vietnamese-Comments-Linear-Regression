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
	#comments = []
	star_1 = []
	star_2 = []
	star_3 = []
	star_4 = []
	star_5 = []
	for rows in range(1,sheet.nrows): 
		star = sheet.cell_value(rows,1)
		if star <= 1:
			star_1.append( sheet.cell_value(rows,0) )
		elif star <= 2:
			star_2.append( sheet.cell_value(rows,0) )
		elif star <= 3:
			star_3.append( sheet.cell_value(rows,0) )
		elif star <= 4:
			star_4.append( sheet.cell_value(rows,0) )
		else:
			star_5.append( sheet.cell_value(rows,0) )
	stars = []
	stars.append(star_1)
	stars.append(star_2)
	stars.append(star_3)
	stars.append(star_4)
	stars.append(star_5)
	for i in range (len(stars)):
		print ('len(star_{}) = {}'.format(i+1, len(stars[i])))
	return stars
	
def split(stars, train_len = 1000, test_len = 50):
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	for i in range (len(stars)):
		X_train += stars[i][:train_len]
		y_train += [(i+1) for t in range (train_len)]
		X_test += stars[i][-test_len:]
		y_test += [(i+1) for t in range (test_len)]
	return X_train, y_train, X_test, y_test
	
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
	print ('Len(BoW) = {}'. format(len(bag_of_word)))
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
			for word in X[i]:
				try:
					if word in index_dict.keys():
						col_id.append(index_dict[word])
						row_id.append(i)
						data_id.append(1)
				except KeyError:
					bug = False
		new_X = coo_matrix((data_id, (row_id, col_id)), shape=(len(X), len(bag_of_word)))
		return new_X
	
	new_X_train = feature_extract(X_train)
	new_X_test = feature_extract(X_test)
	return new_X_train, new_X_test
		
def main():
	print ('Reading from xls file')
	stars = read_xls()
	print ('Read successfully')
	print ('Features extracting')
	X_train, y_train, X_test, y_test = split(stars)
	X_train, X_test = to_vector(X_train, X_test)
	print ('Extracted features successfully')
	
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
	