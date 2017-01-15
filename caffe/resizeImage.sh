cd ASD
for a in *.png; 
do 
echo $a
convert -resize 256x256\! $a $a
done

cd ..

cd TD

for a in *.png; 
do
echo $a 
convert -resize 256x256\! $a $a
done

cd ..

