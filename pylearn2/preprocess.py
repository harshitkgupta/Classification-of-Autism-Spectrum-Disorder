import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
import scipy 
X=pd.read_csv('data.csv',index_col=[0])
print X.shape
X = preprocessing.scale(X)
#X=np.array(X)
def stft(x, fftsize=512, overlap_pct=.5):   
  hop = int(fftsize * (1 - overlap_pct))
  w = scipy.hanning(fftsize + 1)[:-1]    
  raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
  return raw[:, :(fftsize / 2)]
#Y=[]        
#for i in range(X.shape[0]):
#  d = np.abs(stft(X[i, :-1]))        
#  Y.append(d);   

np.random.seed(seed=0)
Y=X
for i in range(0,150):
	Y[i]=X[i];
	Y[2*i+1]=X[150+i];


X=pd.DataFrame(X)
print X.shape
TEST_FRAC = 0.15
VALID_FRAC = 0.15
test_set_size = int(TEST_FRAC * len(X))
valid_set_size = int(VALID_FRAC * len(X))

X=X.iloc[np.random.permutation(len(X))]

X[0:test_set_size].to_csv("test.csv", index=False)
X[test_set_size:test_set_size+valid_set_size].to_csv("valid.csv", index=False)
X[test_set_size+valid_set_size:].to_csv("train.csv", index=False)
#pickle.dump(np.array(X[0:test_set_size][:-1]), open('saved_tst.pkl', 'wb'))
