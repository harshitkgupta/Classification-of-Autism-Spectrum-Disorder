from scipy.signal import butter, lfilter,filtfilt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import wave

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    print len(a),len(b),len(data)
    y = lfilter(b, a, np.array(data).T)
    return y

def BandPassFilter(audiofile):

	lowcut = 300.0
    	highcut = 3400.0
	spf=wave.open(audiofile,'rb')
	fs=spf.getframerate()
	x = spf.readframes(-1)
	CHANNELS=1
        swidth=2
        spf.close()
        y = butter_bandpass_filter(x, lowcut, highcut, fs, order=10)
	wf = wave.open(audiofile, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(swidth)	
	wf.writeframes(y)
	wf.close()
		


    
    

