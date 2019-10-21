clear all
close all

% this script generates input data, makes plots to visualize the classes of
% the data

load waveform.data;

X = waveform(:,1:21);
y = waveform(:,22);


[n d]=size(X);
t=1:d;

K = max(y);
if min(y)==0; 
    K=K+1;
    y = y+1; 
end

color={'r','g','b','k','c','m','y'};
figure
for k=1:K
    hold on
    plot(1:d,X(y==k,:)',color{k});
end
xlim([1 d])
box on
title('Waveform data');






