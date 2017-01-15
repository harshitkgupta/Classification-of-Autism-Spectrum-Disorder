from pydub import AudioSegment
size=10;
for i in range(1,size+1):
	filename='ASD/f'+str(i)+'.wav';
	sound = AudioSegment.from_wav(filename);
	sound = sound.set_channels(1);
	sound.export(filename, format="wav");

for i in range(1,size+1):
        filename='TD/f'+str(i)+'.wav';
        sound = AudioSegment.from_wav(filename);
        sound = sound.set_channels(1);
        sound.export(filename, format="wav");

for i in range(1,size+1):
        filename='DD/f'+str(i)+'.wav';
        sound = AudioSegment.from_wav(filename);
        sound = sound.set_channels(1);
        sound.export(filename, format="wav");
