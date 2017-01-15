from SamplingRate import DoubleSamplingRate

size=10;
for i in range(1,size+1):
	filename='ASD/f'+str(i)+'.wav';
	DoubleSamplingRate(filename);
	

for i in range(1,size+1):
        filename='TD/f'+str(i)+'.wav';
        DoubleSamplingRate(filename);
       

for i in range(1,size+1):
        filename='DD/f'+str(i)+'.wav';
        DoubleSamplingRate(filename);
        

