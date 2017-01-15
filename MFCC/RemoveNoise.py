import BandPassFilter as bp

size=10;
for i in range(1,size+1):
	filename='ASD/f'+str(i)+'.wav';
	bp.BandPassFilter(filename);
	

for i in range(1,size+1):
        filename='TD/f'+str(i)+'.wav';
        bp.BandPassFilter(filename);
       

for i in range(1,size+1):
        filename='DD/f'+str(i)+'.wav';
        bp.BandPassFilter(filename);
        

