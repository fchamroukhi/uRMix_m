
clear all
close all

load npfda-phoneme.dat;
% x contient les observations
x = npfda_phoneme(:,1:150);

% y contient les labels des observations
y = npfda_phoneme(:,151);


[n d]=size(x)

% x1, x2, x3, x4 et x5 contiennent chacune les donnÃ©es correspendantes aux
% classes des phonemes respectifs â€œshâ€?, "iy", â€œdclâ€?, "aa" et "ao"
x1 = x(y==1,:);
x2 = x(y==2,:);
x3 = x(y==3,:);
x4 = x(y==4,:);
x5 = x(y==5,:);

figure(1);

% pour mieux visualiser les classes, on ne reprensente que 20 observations
% de chaque classe
%hold on;
%subplot(511);
plot(1:d,x1(1:20,:),'b');
title('sh')
%subplot(512);
figure; plot(1:d,x2(1:20,:),'b');
title('iy')
%subplot(513);
figure; plot(1:d,x3(1:20,:),'b');
title('dcl')
%subplot(514);
figure; plot(1:d,x4(1:20,:),'b');
title('aa')
%subplot(515);
figure; plot(1:d,x5(1:20,:),'b');
title('ao')
hold off;


X = x;
[n d] = size(X);
K = max(y);
figure
title('Phoneme data')
for k=1:K
    hold on
    subplot(3,2,k)
    Xk = X(y==k,:); 
    plot(1:d,Xk(1:50,:),'b');
end


