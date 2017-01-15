__author__ = 'hk'

import sys
import csv
import wave
import numpy as np
import scipy as sp
import pywt
from scipy.io import wavfile	
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut,StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.ensemble import  RandomForestClassifier
from sklearn import preprocessing
import cPickle
import os
asd=[]

td=[]
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
	level=15;
	wname='db8';
	print "extracting features -----"
	print "wavelet name   ",wname,"  level  ",level
	for i in xrange(0,asd_n):
		print "asd ",i	
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
		print "td ",i	
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
	X=np.matrix(X)
	Y=np.array(Y);
        
    
	X = preprocessing.scale(X)
	print X.shape,Y.shape
	dlname='rr1.pkl'
	if not os.path.exists('./%s'%dlname):
		""" Trains and predicts dataset with a Random forest classifier """
		cvk = LeaveOneOut(len(X))
		#cvk=StratifiedKFold(Y,n_folds=5)
	    	clf =RandomForestClassifier();
	    	no_trees=[10,30,50,70]
	   	param = dict(n_estimators=no_trees,n_jobs=[-1])
	    	clf = GridSearchCV(clf,param_grid=param,cv=cvk)
	    	clf.fit(X,Y)
	    	s = cPickle.dump(clf.best_estimator_,file(dlname, 'wb'))
	else:
		print "loading random forest model"
		clf=cPickle.load(file(dlname, 'rb'))    	
    	
    	
   	print("The best classifier is: ",clf)
    	# Estimate score
    	scores = cross_validation.cross_val_score(clf, X,Y)
    	print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    	#Predict and save
    


if __name__ == "__main__":
   
    asd,td= read_datasets();

    xtrain,ytrain=extract_features(asd,td);
    
    train_test_model(xtrain, ytrain);
    
