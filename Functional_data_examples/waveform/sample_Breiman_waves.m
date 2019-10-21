function [X klas] = sample_Breiman_waves(n)

% by Faicel Chamroukhi

K=3;
Pik = 1/K*ones(K,1);%mixing proportions

dt = 1;%periode d'echantilonnage (secondes)
fs = 1/dt;
temps = 0:dt:21-dt; %temps;
m = length(temps);

X = zeros(n,m);
klas = zeros(n,1);



h0 = max(6-abs(temps-11),0);

Dt = 4/dt;

h1 = [zeros(1,Dt) h0(1:end-Dt)];
h2 = [h0(Dt+1:end) zeros(1,Dt)];


for i=1:n
    
    mu = rand(1,m);
    ei = randn(1,m); 
    %
    zik = mnrnd(1,Pik);
    zi = find(zik==1);
    klas(i) = zi;
    if zi==1
        X(i,:)=mu.*h0 + (1 -mu).*h1 + ei;
    elseif zi==2
        X(i,:) = mu.*h0 + (1 -mu).*h2 + ei;
    elseif zi==3
        X(i,:) = mu.*h1 + (1 -mu).*h2 + ei;
    else error('unknown cluster label'); 
    end
end
    