cd ASD
for p in *.wav; 
do 
normalize-audio -a --$p; 
done

cd ..

cd TD

for p in *.wav; 
do 
normalize-audio -a --$p; 
done

cd ..
