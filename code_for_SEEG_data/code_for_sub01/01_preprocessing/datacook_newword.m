clear all
close all

%% load
Ada1=pop_biosig( 'Adaptation1.edf');
Ada2=pop_biosig( 'Adaptation2.edf');
Ada3=pop_biosig( 'Adaptation3.edf');
Ada4=pop_biosig( 'Adaptation4.edf');
dataCell = {Ada1.data, Ada2.data, Ada3.data, Ada4.data};
mergedData = [];
for i = 1:4
    mergedData = [mergedData, dataCell{i}];
end
EEG1 = Ada1;
EEG1.data = mergedData;
EEG1.pnts = size(mergedData,2);
EEG1.xmax = EEG1.xmin + (EEG1.pnts-1)/EEG1.srate;
EEG1 = eeg_checkset(EEG1);

%% delect bad electrodes
EEG2 = pop_chanevent( EEG1, [ ],'edge','leading','oper','X > 10000' );   

%% plot trigger channel to check
chan_trig = EEG2.nbchan; %127;
figure;
plot( EEG2.data(chan_trig,:)); 
% close

%% modify trigger 
EEG3 = EEG2;

%% read trigger 
posEEG = pop_chanevent(  EEG3, [ EEG2.nbchan],'edge','leading','oper','X > 2000' );

%% delete some wrong
   for i = 2:  length(posEEG.event)
     delLan=posEEG.event(i).latency-posEEG.event(i-1).latency;
        if delLan < 200
            posEEG.event(i).type = num2str(999);
            posEEG.urevent(i).type = num2str(999);
        else
           
    end
   end
   
j=0;
  for i = 1:length(posEEG.event)
      if str2num(posEEG.event(i).type)==1
            j=j+1;
      else
      end
  end
 disp('num_event = ');
 num_event = j

%% plot spectrum

signal = posEEG.data(3,:); 
Fs = posEEG.srate;        

Y = fft(signal);
L = length(signal);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

figure;
plot(f, P1);
title('Channel 1 Spectrum (Quick Version)');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
grid on;


%% broad-band
 [posEEG0, com, b] = pop_eegfiltnew( posEEG, 0.5, 200);  

%% narrow band
% [posEEG0, com, b] = pop_eegfiltnew(posEEG_notch,4,8); 
% [posEEG0, com, b] = pop_eegfiltnew(posEEG_notch,8,12); 
% [posEEG,0 com, b] = pop_eegfiltnew(posEEG_notch,12,30); 
% [posEEG0, com, b] = pop_eegfiltnew(posEEG_notch,30,60); 
[posEEG0, com, b] = pop_eegfiltnew(posEEG,60,150); 
    for i=1:size(posEEG0.data,1)
     posEEG0.data(i,:)=abs(hilbert(posEEG0.data(i,:)));
    end
   
%% load logfile
load logfile_newword   %什么刺激

%% type = = condition
j=1;
  for i = 1:length(posEEG.event)
      if str2num(posEEG.event(i).type)==1
            posEEG0.event(i).type = num2str(logfile(1,j));
            posEEG0.urevent(i).type = num2str(logfile(1,j));
            j=j+1;
      else
      end
  end
 
%% epoch

a = -0.2;
b = 1.2;

wordNames = { ...
    '草莓','带鱼','蛋糕','豆腐','海带','红薯','鸡蛋','煎饼', ...
    '荔枝','龙虾','萝卜','绿豆','芒果','蜜桔','面包','蘑菇', ...
    '牛肉','苹果','软糖','薯条','西瓜','香肠','洋葱','樱桃', ...
    '玉米','猪蹄','草坪','代表','弹弓','逗号','海滩','红灯', ...
    '机会','肩膀','力量','笼子','螺旋','律师','盲人','密码', ...
    '面具','模特','牛仔','评委','软件','暑假','膝盖','乡村', ...
    '阳光','英雄','浴室','珠宝'};

ALLEEG = repmat(posEEG0, 1, 52);

for i = 1:52

    ALLEEG(i) = pop_epoch(posEEG0, {num2str(i)}, [a b]);

    ALLEEG(i).setname = wordNames{i};

end

