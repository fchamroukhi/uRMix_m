clear all
close all

load npfda-sat.dat;

X = npfda_sat;
[n d] = size(X);
figure;
plot(X','b');
box on
title('Satellite data');
