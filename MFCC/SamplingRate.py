import wave

def DoubleSamplingRate(audiofile):
	Change_rate=2
	CHANNELS = 1
	swidth = 2
	spf=wave.open(audiofile,'rb')
	RATE=spf.getframerate()
	signal = spf.readframes(-1)
	spf.close()
	if RATE==22050:
		wf = wave.open(audiofile, 'wb')
		wf.setnchannels(CHANNELS)
		wf.setsampwidth(swidth)
		wf.setframerate(RATE*Change_rate)
		wf.writeframes(signal)
		wf.close()
		print "changed sampling rate of ",audiofile 
