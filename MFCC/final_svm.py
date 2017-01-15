__author__ = 'hk'

import sys
import csv

import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.cross_validation import StratifiedKFold, train_test_split,LeaveOneOut
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.mixture import GMM
from sklearn.qda import QDA
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


def read_datasets():
    """ Reads test and training csv files """
    # Open the data files
    # TODO setup error handling for this in case file not present
    data_file = open('data.csv','r')

    X=[] 
    Y=[]
    
    # Read in CSV file
   
    reader = csv.reader(data_file, delimiter=',')
    header = reader.next()
    feature=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26,27,29,30,31,32,33,34,35,36]
    features=[3,7,12,13,14,31,33,34]
    for row in reader:
	if row[3]=='M':
		row[3]=1
	else:
		row[3]=2
	d=[]
	for i in features:
		d.append(row[i])
    	X.append(d)
        Y.append(row[1])
            
    
    # Close the files 
    data_file.close()
    
    # Return Xtrain, Ytrain, and Xtest
    return X, Y





def trainModel(X, Y ):
    """ Trains and predicts dataset with a SVM classifier """
    c_range = 10.0 ** np.arange(-2, 9)
    gamma_range = 10.0 ** np.arange(-5, 4)
    param = [{'kernel': ['rbf']}]

    
    
    cvk = LeaveOneOut(len(X))
    classifier = SVC()

    clf = GridSearchCV(classifier,param_grid=param,cv=cvk)
    clf.fit(X,Y)
    print("The best classifier is: ",clf.best_estimator_)
    # Estimate score
    scores = cross_validation.cross_val_score(clf.best_estimator_, X,Y)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    # Predict and save
    result = clf.best_estimator_.predict(X)



    


if __name__ == "__main__":
    
    # This is a pretty small dataset 
    # It will load pretty quickly.
    X, Y = read_datasets()
    # Train random forest, predict result, write to output
    trainModel(X, Y)
    #write_test_labels(Ytest, outfile=predictionsFile)
