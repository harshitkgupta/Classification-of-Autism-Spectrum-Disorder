import numpy as np
import deepdish as dd
import lmdb
import caffe
import wave

dataset=20
data=[]
y=[]
l=0;	
for i in xrange(0,dataset):
	filename='ASD/f'+str(i+1)+'.wav';
	sf,x=wavfile.read(filename)
	if x.ndim>1:
		x=np.mean(x,axis=1);
	l=max(len(x),l)		
        data.append(x);

for i in xrange(0,dataset):
	filename='TD/f'+str(i+1)+'.wav';
       	sf,x=wavfile.read(filename)
        if x.ndim>1:
		x=np.mean(x,axis=1);
	l=max(len(x),l)	
        data.append(x);
	

N = 1000


X = np.zeros((N, 1, 1, l), dtype=np.uint16)
y = np.array(y)

# We need to prepare the database for the size. If you don't have 
# deepdish installed, just set this to something comfortably big 
# (there is little drawback to settings this comfortably big).
map_size = dd.bytesize(X) * 2

env = lmdb.open('mylmdb', map_size=map_size)

for i in range(N):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = X.shape[1]
    datum.height = X.shape[2]
    datum.width = X.shape[3]
    datum.data = X[i].tobytes()
    datum.label = int(y[i])
    str_id = '{:08}'.format(i)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

