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
from sklearn.ensemble import  RandomForestClassifier
from sklearn.mixture import GMM
from sklearn import preprocessing
import scipy.io as sio

from GetFeatures import ReadWavFiles



dataset=10

def read_datasets():
    """ Reads test and training files """
    # Open the data files
    # TODO setup error handling for this in case file not present
    Xtrain=[]
    Ytrain=[]
    Xtest=[]
    
    asd,dd,td=ReadWavFiles(dataset,True);	
    mfcclim=40000
    mfcclim_d1=40000
    mfcclim_d2=40000
    aclim=40000	
    for i in xrange(0,dataset):
    	mfcclim=min(mfcclim,asd[i]['mfcc'].shape[0])
	mfcclim=min(mfcclim,dd[i]['mfcc'].shape[0])
	mfcclim=min(mfcclim,td[i]['mfcc'].shape[0])
	mfcclim_d1=min(mfcclim_d1,asd[i]['mfcc_d1'].shape[0])
	mfcclim_d1=min(mfcclim_d1,dd[i]['mfcc_d1'].shape[0])
	mfcclim_d1=min(mfcclim_d1,td[i]['mfcc_d1'].shape[0])
        
	mfcclim_d2=min(mfcclim_d2,asd[i]['mfcc_d2'].shape[0])
	mfcclim_d2=min(mfcclim_d2,dd[i]['mfcc_d2'].shape[0])
	mfcclim_d2=min(mfcclim_d2,td[i]['mfcc_d2'].shape[0])		
        
	lpclim=min(aclim,asd[i]['lpc'].shape[0])
	lpclim=min(aclim,dd[i]['lpc'].shape[0])
	lpclim=min(aclim,td[i]['lpc'].shape[0])
         
    for i in xrange(0,dataset):
	d=[]
        d.append(np.array(asd[i]['mfcc'][:mfcclim][:]).ravel());
        #d.append(np.array(asd[i]['mfcc_d1'][:mfcclim_d1][:]).ravel());
	#d.append(np.array(asd[i]['mfcc_d2'][:mfcclim_d2][:]).ravel());
#d.append(np.array(asd[i]['lpc'][:lpclim][:]).ravel());
        
        Xtrain.append(np.asarray(d).ravel());
        Ytrain.append(1);

    for i in xrange(0,dataset):
	d=[]
	d.append(np.array(dd[i]['mfcc'][:mfcclim][:]).ravel());
        #d.append(np.array(dd[i]['mfcc_d1'][:mfcclim_d1][:]).ravel());
	#d.append(np.array(asd[i]['mfcc_d2'][:mfcclim_d2][:]).ravel());
	#d.append(np.array(asd[i]['lpc'][:lpclim][:]).ravel());        
	Xtrain.append(np.asarray(d).ravel());
        Ytrain.append(0);


    for i in xrange(0,dataset):
	d=[]
	d.append(np.array(td[i]['mfcc'][:mfcclim][:]).ravel());
 #       d.append(np.array(td[i]['mfcc_d1'][:mfcclim_d1][:]).ravel());
#d.append(np.array(asd[i]['mfcc_d2'][:mfcclim_d2][:]).ravel());
        #d.append(np.array(asd[i]['lpc'][:lpclim][:]).ravel());
        Xtrain.append(np.asarray(d).ravel());
        Ytrain.append(-1);

    Xtrain=np.array(Xtrain)
    
    # Return Xtrain, Ytrain, and Xtest
    return Xtrain, Ytrain


def write_test_labels(Ytest, outfile="data/testLabels.csv"):
    """ Writes 9000 testing predictions to file """
    f = open(outfile, 'w')
    f.write('Id,Solution\n')
    count = 1
    for prediction in Ytest:
        f.write("%d,%d\n" % (count, prediction))
        count += 1
    f.close()


def train(X, Y):
    """ Trains and predicts dataset with a RandomForest classifier """
   
    

    cvk = LeaveOneOut(len(X))
    clf =RandomForestClassifier();
    no_trees=[10,30,50]
    param = dict(n_estimators=no_trees,n_jobs=[-1])
    clf = GridSearchCV(clf,param_grid=param,cv=cvk)
    clf.fit(X,Y)
    #clf.fit(X,Y)
    print("The best classifier is: ",clf.best_estimator_)
    # Estimate score
    scores = cross_validation.cross_val_score(clf.best_estimator_, X,Y,cv=cvk,n_jobs=-1)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    #Predict and save
    


if __name__ == "__main__":
    # Argument handling for custom output files
    
    # This is a pretty small dataset 
    # It will load pretty quickly.
    Xtrain, Ytrain = read_datasets()
    # Train random forest, predict result, write to output
    train(Xtrain, Ytrain)
    #print accuracy_score(Ytrain,Ytest)
    #write_test_labels(Ytest, outfile=predictionsFile)
