from Features import ExtractFeatures
import pickle

def ReadWavFiles(size=10,out=True):
	asd=[]
	dd=[]
	td=[]
        outfile='featers.csv'
	if not out:
		with open(outfile,'rb') as handle:
			pickle.load(asd,handle)
			pickle.load(dd,handle)
 			pickle.load(td,handle)
		return (asd,dd,td) 
	for i in range(1,size+1):
		filename='ASD/f'+str(i)+'.wav';
		features = ExtractFeatures(filename);
		asd.append(features)

	for i in range(1,size+1):
		filename='TD/f'+str(i)+'.wav';
		features = ExtractFeatures(filename);
		td.append(features)

	for i in range(1,size+1):
		filename='DD/f'+str(i)+'.wav';
		features= ExtractFeatures(filename);
		dd.append(features)	

	#engine.writeInput(outfile,np.array(feats))
	
	
	with open(outfile,'wb') as handle:
		pickle.dump(asd,handle)
		pickle.dump(dd,handle)
 		pickle.dump(td,handle) 
	
	return(asd,dd,td)
