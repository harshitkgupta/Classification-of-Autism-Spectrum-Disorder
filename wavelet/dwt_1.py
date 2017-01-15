import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 28}

#plt.rcParams.update({'font.size': 40})

width = 0.20
N = 5
fig, ax = plt.subplots()
ind = np.arange(N)

haar = (73.1,67.2,71.6,68.7,68.7)
rects1 = ax.bar(ind, haar, width, color='r')

db4 = (68.6,67.1,67.3,67.14,65)
rects2 = ax.bar(ind+width, db4, width, color='g')

db8=(64.4,71.8,70.1,67.25,68.57)
rects3 = ax.bar(ind+2*width, db8, width, color='b')

db16=(68.5,76.2,73.4,74.6,74.6)
rects4 = ax.bar(ind+3*width, db16, width, color='#4B0082')

ax = plt.gca()

ax.set_ylim([60,85])
# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy',fontsize=24)
ax.set_xlabel('Decomposition Level',fontsize=24)
ax.set_title('Cross-validation scores for DWT',fontsize=24)
ax.set_xticks(ind+width)
ax.set_xticklabels( ('4', '5', '6', '7', '8') ,fontsize=24)

ax.legend( (rects1[0], rects2[0],rects3[0],rects4[0]), ('haar', 'db4','db8','db16') )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()	
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom',fontsize=24)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)


plt.show()