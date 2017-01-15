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
from sklearn import preprocessing
import scipy.io as sio

from GetFeatures import ReadWavFiles

asd=[]
dd=[]
td=[]
dataset=10

def read_datasets():
    """ Reads test and training files """
    # Open the data files
    # TODO setup error handling for this in case file not present
    Xtrain=[]
    Ytrain=[]

    asd,dd,td=ReadWavFiles(dataset)	
    mfcclim=40000
    mfcclim_d1=40000	
    for i in xrange(0,dataset):
        mfcclim=min(mfcclim,asd[i]['mfcc'].shape[0])
        mfcclim=min(mfcclim,dd[i]['mfcc'].shape[0])
        mfcclim=min(mfcclim,td[i]['mfcc'].shape[0])
        mfcclim_d1=min(mfcclim_d1,asd[i]['mfcc_d1'].shape[0])
        mfcclim_d1=min(mfcclim_d1,dd[i]['mfcc_d1'].shape[0])
        mfcclim_d1=min(mfcclim_d1,td[i]['mfcc_d1'].shape[0])		
    
    for i in xrange(0,dataset):
	d=[]
        d.append(np.array(asd[i]['mfcc'][:mfcclim][:]).ravel());
        d.append(np.array(asd[i]['mfcc_d1'][:mfcclim_d1][:]).ravel());
        Xtrain.append(np.asarray(d).ravel());
        Ytrain.append(1);

  #  for i in xrange(0,dataset):
	#d=[]
  #      d.append(np.array(dd[i]['mfcc'][:mfcclim][:]).ravel());
  #      d.append(np.array(dd[i]['mfcc_d1'][:mfcclim_d1][:]).ravel());
  #      Xtrain.append(np.asarray(d).ravel());
  #      Ytrain.append(0);


    for i in xrange(0,dataset):
	d=[]
        d.append(np.array(td[i]['mfcc'][:mfcclim][:]).ravel());
        d.append(np.array(td[i]['mfcc_d1'][:mfcclim_d1][:]).ravel());
        Xtrain.append(np.asarray(d).ravel());
        Ytrain.append(-1);






    # Return Xtrain, Ytrain, and Xtest
    return np.array(Xtrain), Ytrain

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

        # Decomposition
   # X_all=np.r_[Xtrain,Xtest]
    pca = PCA( whiten=True)
    #X_all=pca.fit_transform(X_all)
    X = pca.fit_transform(X)
    #Xtest = pca.transform(Xtest)


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
    


if __name__ == "__main__":
    # Argument handling for custom output files

    
    # This is a pretty small dataset 
    # It will load pretty quickly.
    Xtrain, Ytrain= read_datasets()
    # Train random forest, predict result, write to output
    train(Xtrain, Ytrain)
    #print accuracy_score(Ytrain,Ytest)
    #write_test_labels(Ytest, outfile=predictionsFile)
