__author__ = 'hk'

import sys
import csv
import wave
import numpy as np
import scipy as sp
import pywt
from scipy.io import wavfile	
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut,StratifiedKFold,KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import cPickle
import os
import scipy.io as sio	
asd_n=40
td_n=27

def read_datasets():
        """ Reads test and training files """
	asd=[]
	td=[]
		
	for i in xrange(0,asd_n):
		filename='ASD/f'+str(i+1)+'.wav';
		sf,x=wavfile.read(filename)
		if x.ndim>1:
			x=np.mean(x,axis=1);	
        	asd.append(x);

	for i in xrange(0,td_n):
		filename='TD/f'+str(i+1)+'.wav';
        	sf,x=wavfile.read(filename)
        	if x.ndim>1:
			x=np.mean(x,axis=1);	
        	td.append(x);	

	return asd,td

def extract_features(asd,td):
	trainX=[]
	trainY=[]
	#level=5;
	#wname='db16';
	level=int(raw_input());
	wname=raw_input();
	print "extracting features -----"
	print "wavelet name   ",wname,"  level  ",level
	for i in xrange(0,asd_n):
		#print "asd ",i	
		x_asd=asd[i]
		coef= pywt.wavedec(x_asd, wname, level=level);
		
		f1=[]
		f1.append(np.mean(x_asd));		
		for cf in coef:
			f1.append(np.mean(cf));

		f2=[]
		f2.append(np.std(x_asd));
		for cf in coef:	
			f2.append(np.std(cf));
       		
		f3=[]
		for j in xrange(1,len(f1)):			
			f3.append(f1[j-1]-f1[j]);
			
		
		f4=[]
		for j in xrange(1,len(f2)):
			f4.append(f2[j-1]-f2[j]);

		f5=[]
		for j in xrange(1,len(f1)-1):
			f5.append(f1[j-1]-2*f1[j]+f1[j+1]);
		
		f6=[]
		for j in xrange(1,len(f2)-1):
			f6.append(f2[j-1]-2*f2[j]+f2[j+1]);
		
		f=f1+f2+f3+f4+f5+f6;
		trainX.append(f);
		trainY.append(1);

	for i in xrange(0,td_n):
		#print "td ",i	
		x_td=td[i]
		coef= pywt.wavedec(x_td, wname, level=level)

		f=[]

		f1=[]
		f1.append(np.mean(x_td));		
		for cf in coef:
			f1.append(np.mean(cf));

		f2=[]
		f2.append(np.std(x_td));
		for cf in coef:	
			f2.append(np.std(cf));
       		
		f3=[]
		for j in xrange(1,len(f1)):			
			f3.append(f1[j-1]-f1[j]);
			
		
		f4=[]
		for j in xrange(1,len(f2)):
			f4.append(f2[j-1]-f2[j]);

		f5=[]
		for j in xrange(1,len(f1)-1):
			f5.append(f1[j-1]-2*f1[j]+f1[j+1]);
		
		f6=[]
		for j in xrange(1,len(f2)-1):
			f6.append(f2[j-1]-2*f2[j]+f2[j+1]);
		
		f=f1+f2+f3+f4+f5+f6;
		trainX.append(f);
		
		trainY.append(0);

	

	return trainX,trainY;

def train_test_model(X, Y):
	
	
	
	
	X=np.matrix(X);
	Y=np.array(Y);
        # Decomposition   	
	#pca = PCA(whiten=True)
   	#X = pca.fit_transform(X)

    	#scaling 
	X = preprocessing.scale(X)
	
	clf=None
	#X,Y = shuffle(X,Y)
	print X.shape,Y.shape
	dlname='svm1.pkl'
	#cvk=StratifiedKFold(Y,n_folds=5)
	cvk = LeaveOneOut(len(X))
	if not os.path.exists('./%s'%dlname):
		""" Trains and predicts dataset with a SVM classifier """
		c_range =10**np.arange(1, 5,0.25)
		gamma_range = 10.0 ** np.arange(-4,4,0.25)
		param = [{'kernel': ['rbf'],'gamma':gamma_range,'C':c_range}]

	    
	    
		
		
		classifier = SVC()
		#print 'cross validator',cvk
		clf = GridSearchCV(classifier,param_grid=param,cv=cvk)
		clf.fit(X,Y)
		s = cPickle.dump(clf,file(dlname, 'wb'))
		
	else:
		print "loading svm model"
		clf=cPickle.load(file(dlname, 'rb'))	
        clf.fit(X,Y)
	
        print("The best classifier is: ",clf.best_estimator_)

	
        # Estimate score
        scores = cross_validation.cross_val_score(clf.best_estimator_, X,Y,cv=cvk)
        print scores
        print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
	cm = confusion_matrix(Y ,clf.best_estimator_.predict(X))
	print cm
        # Predict and save
    


if __name__ == "__main__":
   
    asd,td= read_datasets();

    xtrain,ytrain=extract_features(asd,td);
    
    train_test_model(xtrain, ytrain);
    
