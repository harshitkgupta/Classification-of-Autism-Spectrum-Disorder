clear all;
close all;
s=20

for i=1:s
	
	
	infile=strcat('ASD/f',num2str(i),'.wav');
	[f,Fs] = wavread(infile);
	%% Design a bandpass filter that filters out between 700 to 12000 Hz
	n = 7;
	beginFreq = 300 /(Fs/2);
	endFreq = 3400/(Fs/2) ;
	[b,a] = butter(n, [beginFreq, endFreq]);
	%% Filter the signal
	fOut = filter(b, a, f);
	outfile=strcat('ASD/F',num2str(i));	
	fh=figure;
        x=mean(fOut,2);     
        step = fix(5*Fs/1000);    
        window = fix(40*Fs/1000);  
        fftn = 2^nextpow2(window);               
        [S, f, t] = specgram(x, fftn, Fs, window, window-step);              
        S = abs(S(2:fftn*4000/Fs,:)); 
        S = S/max(S(:));           
        S = max(S, 10^(-40/10)); 
        S = min(S, 10^(-3/10));
        %colormap bone;    
        imagesc(t, f, flipud(log(S)));   
        set(gcf,'Visible','off')              % turns current figure "off"
	set(0,'DefaultFigureVisible','off');                       
  	set(gca,'XTick',[]) % Remove the ticks in the x axis!
	set(gca,'YTick',[]) % Remove the ticks in the y axis
	set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
	saveas(gcf,outfile,'png')
	
end	





for i=1:s
	;
	infile=strcat('TD/f',num2str(i),'.wav');
	[f,Fs] = wavread(infile);
	%% Design a bandpass filter that filters out between 700 to 12000 Hz
	n = 7;
	beginFreq = 300 /(Fs/2);
	endFreq = 3400/(Fs/2) ;
	[b,a] = butter(n, [beginFreq, endFreq]);
	%% Filter the signal
	fOut = filter(b, a, f);
	outfile=strcat('TD/F',num2str(i));	
	fh=figure;
        x=mean(fOut,2);      
        step = fix(5*Fs/1000);    
        window = fix(40*Fs/1000);  
        fftn = 2^nextpow2(window);               
        [S, f, t] = specgram(x, fftn, Fs, window, window-step);              
        S = abs(S(2:fftn*4000/Fs,:)); 
        S = S/max(S(:));           
        S = max(S, 10^(-40/10)); 
        S = min(S, 10^(-3/10));    
        %colormap bone;    
        imagesc(t, f, flipud(log(S)));   
        set(gcf,'Visible','off')              % turns current figure "off"
	set(0,'DefaultFigureVisible','off');                          
  	set(gca,'XTick',[]) % Remove the ticks in the x axis!
	set(gca,'YTick',[]) % Remove the ticks in the y axis
	set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
	saveas(gcf,outfile,'png')
	
end	
