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
import scipy.io.wavfile
import scipy.io as sio
import numpy as np
import math, cmath
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import lfilter
import melfeat


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
    Xtest=[]

    for i in xrange(0,dataset):
        asd.append(sio.loadmat('ASD/f'+str(i+1)+'.mat'));

    for i in xrange(0,dataset):
        dd.append(sio.loadmat('DD/f'+str(i+1)+'.mat'));

    for i in xrange(0,dataset):
        td.append(sio.loadmat('TD/f'+str(i+1)+'.mat'));

    for i in xrange(0,dataset):
        lim = asd[i]['nb']*2 -1;
        sr= asd[i]['stim_stat_zero'][:lim][:,300:311];
        ft=asd[i]['stim_stat_zero'][:lim][:,312:351];
        poa=asd[i]['stim_stat_zero'][:lim][:,352:401];
        d=[];
        d.append(sum(map(sum, [sum(x) for x in zip(sr, sr)])));
        d.append(sum(map(sum, [sum(x) for x in zip(ft, ft)])));
        d.append(sum(map(sum, [sum(x) for x in zip(poa, poa)])));
        Xtrain.append(d);
        Ytrain.append(1);

    for i in xrange(0,dataset):
        lim = dd[i]['nb']*2 -1;
        sr= dd[i]['stim_stat_zero'][:lim][:,300:311];
        ft=dd[i]['stim_stat_zero'][:lim][:,312:351];
        poa=dd[i]['stim_stat_zero'][:lim][:,352:401];
        d=[];
        d.append(sum(map(sum, [sum(x) for x in zip(sr, sr)])));
        d.append(sum(map(sum, [sum(x) for x in zip(ft, ft)])));
        d.append(sum(map(sum, [sum(x) for x in zip(poa, poa)])));
        Xtrain.append(d);
        Ytrain.append(0);


    for i in xrange(0,dataset):
        lim = td[i]['nb']*2 -1;
        sr= td[i]['stim_stat_zero'][:lim][:,300:311];
        ft=td[i]['stim_stat_zero'][:lim][:,312:351];
        poa=td[i]['stim_stat_zero'][:lim][:,352:401];
        d=[];
        d.append(sum(map(sum, [sum(x) for x in zip(sr, sr)])));
        d.append(sum(map(sum, [sum(x) for x in zip(ft, ft)])));
        d.append(sum(map(sum, [sum(x) for x in zip(poa, poa)])));
        Xtrain.append(d);
        Ytrain.append(-1);

    print Xtrain

    # Return Xtrain, Ytrain, and Xtest
    return Xtrain, Ytrain, Xtest


def write_test_labels(Ytest, outfile="data/testLabels.csv"):
    """ Writes 9000 testing predictions to file """
    f = open(outfile, 'w')
    f.write('Id,Solution\n')
    count = 1
    for prediction in Ytest:
        f.write("%d,%d\n" % (count, prediction))
        count += 1
    f.close()


def train(Xtrain, Ytrain, Xtest):
    """ Trains and predicts dataset with a SVM classifier """
    # Decomposition
   # X_all=np.r_[Xtrain,Xtest]
    #pca = PCA(n_components=12, whiten=True)
    #X_all=pca.fit_transform(X_all)
    #Xtrain = pca.transform(Xtrain)
    #Xtest = pca.transform(Xtest)



    clf =RandomForestClassifier(n_estimators=50);


    clf.fit(Xtrain,Ytrain)
    print("The best classifier is: ",clf)
    # Estimate score
    scores = cross_validation.cross_val_score(clf, Xtrain,Ytrain)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    # Predict and save
    result = clf.predict(Xtrain)



    return result


if __name__ == "__main__":
    # Argument handling for custom output files
    print "Data Science London - Scikit Learn Practice"
    print "http://www.kaggle.com/c/data-science-london-scikit-learn"
    print
    print "Usage: python london_data_science.py [Output]"
    predictionsFile = "data/testLabels.csv"
    if len(sys.argv) == 2:
        predictionsFile = sys.argv[1]
        print "Will output predictions to user-specifiied file:", predictionsFile
    else:
        print "No output specified. Will write test predictions to data/testLabels.csv"
    print
    # This is a pretty small dataset - 1000 training, 9000 test.
    # It will load pretty quickly.
    Xtrain, Ytrain, Xtest = read_datasets()
    # Train random forest, predict result, write to output
    Ytest = train(Xtrain, Ytrain, Xtest)
    print accuracy_score(Ytrain,Ytest)
    #write_test_labels(Ytest, outfile=predictionsFile)