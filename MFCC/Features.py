import yaafelib
import numpy as np
import pickle

from yaafelib import FeaturePlan
from yaafelib import Engine
from yaafelib import AudioFileProcessor

def ExtractFeatures(audiofile='zfinch.wav',outfile=None):
	fp = FeaturePlan()
	fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
	fp.addFeature('mfcc_d1: MFCC blockSize=512 stepSize=256 > Derivate DOrder=1')
	fp.addFeature('mfcc_d2: MFCC blockSize=512 stepSize=256 > Derivate DOrder=2')
        fp.addFeature('ac: AutoCorrelation ACNbCoeffs=49  blockSize=1024  stepSize=512')
	fp.addFeature('sr: SpectralRolloff blockSize=512 stepSize=256')
	fp.addFeature('sf: SpectralFlux blockSize=512 stepSize=256')
	fp.addFeature('lpc: LPC LPCNbCoeffs=2  blockSize=1024  stepSize=512')
	engine=Engine()
	df = fp.getDataFlow()
	engine.load(df)
	engine.getInputs()
	engine.getOutputs()


	afp=AudioFileProcessor()
	afp.processFile(engine,audiofile)
	feats=engine.readAllOutputs()
	outfile='feat.csv'
	#engine.writeInput(outfile,np.array(feats))
	
	if outfile is not None:
		with open(outfile,'wb') as handle:
			pickle.dump(feats,handle) 
	return feats



