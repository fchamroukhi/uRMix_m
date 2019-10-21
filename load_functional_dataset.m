function [X klas] = load_functional_dataset(data)
% %%%%%%%% waveform %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(data,'waveform')
    
%     load waveform;
%     n = 50;  klas= [ones(n,1);2*ones(n,1);3*ones(n,1)];
%     X = [waveform.clas1(1:n,:); waveform.clas2(1:n,:) ;waveform.clas3(1:n,:)];

% simulaition

   n = 500;
   [X klas] = sample_Breiman_waves(n); 


% load waveform500;
% X=Y;
% load klasWavfrom500;
end


%%%%%%%%%%% noizy waveform %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(data,'noizy-waveform')
    load('./Functional_data_examples/waveform/waveform-+noise.data');
    X = waveform__noise(:,1:40);
    klas = waveform__noise(:,41)+1;
    %[klas ind] = sort(klas);
    %X = X(ind,:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%% satellite data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(data,'phonemes')
    load ./Functional_data_examples/phonemes/npfda-phoneme.dat;
    % x contient les observations
 % X = npfda_phoneme;
 X = npfda_phoneme(:,1:150);
    % y contient les labels des observations
    klas = npfda_phoneme(:,151);
    
%     %klas = klas(1:100);
%     c1 = klas(klas==1); c1=c1(1:20);
%     c2 = klas(klas==2); c2=c2(1:20);
%     c3 = klas(klas==3); c3=c3(1:20);
%     c4 = klas(klas==4); c4=c4(1:20);
%     c5 = klas(klas==5); c5=c5(1:20);
%     x1 = X(c1,:);
%     x2 = X(c2,:);
%     x3 = X(c3,:);
%     x4 = X(c4,:);
%     x5 = X(c5,:);
%     
% %     x1 = X(klas==1,:);
% %     x2 = X(klas==2,:);
% %     x3 = X(klas==3,:);
% %     x4 = X(klas==4,:);
% %     x5 = X(klas==5,:);
    
    X1=[];
    klas1=[];
    
    nk = 200;% or 400 if we want to consider all the 2000 samples (here we take 1000 samples)
    for k=1:max(klas)
        klask = find(klas==k);
        Xk = X(klask(1:nk),:);
        X1 = [X1;Xk];
        klas1 = [klas1; klas(klask(1:nk))];
    end
    X=X1;
    klas=klas1;
%     for k=1:max(klas)
%         subplot(3,2,k);
%         plot(X(klas==k,:)','b')
%         hold on
%     end
%     pause
end


%%%%%%%%%%%% Tecator data %%%%%%%%%%%%%%%%%%
if strcmp(data,'tecator')
    load ./Functional_data_examples/tecator/tecator.txt
    tecator;
    
    [n m] = size(tecator);
    
    indiceDebut = [1:25:n];
    for i=1:length(indiceDebut)
        Xi = tecator(indiceDebut(i):indiceDebut(i)+19,:)';
        xi = Xi(:);
        X(i,:) = xi;
    end
    klas =[];
%     %X=X(1:120,:);
%     [n m] = size(X);
%     %X = X./(ones(n,1)*max(X)- ones(n,1)*min(X));
%     figure, plot(X','color',[.6 .6 .6])
%     pause
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% gMixReg data (sampled from the model) %%%%%%%%%%
if strcmp(data,'gMixReg_sample')
    %[X true_klas] = sample_gMixReg(Pik,Betak,Sigmak,n,m);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% Polynomial curves %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% toy-nonlin data
if strcmp(data,'toy-nonlin')
%    load Y;
%    X=Y;
% end
% 
% if strcmp(data,'polynomial-curves')
    % clear X
    n1 = 40;
    n2 = 30;
    n3 = 30;
    
    n = n1 + n2 + n3;%
    
    klas = [ones(n1,1);2*ones(n2,1);3*ones(n3,1)];
    
    m = 100;
    x = linspace(0,1,m);
    
    for i=1:n1
        %     Ey1 = 1 + 0.3*exp(-1*x).*sin(1.8*pi*x); % the true function
        Ey1 = .8 + 0.5*exp(-1.5*x).*sin(1.3*pi*x); % the true function
        Ey(:,1) = Ey1;
        %Ey1 = 0.01+ 0.3*exp(-10*x); % the true function
        %Ey=Ey+Ey1;
        sigma = 0.1;
        y = (Ey1 + sigma*randn(size(x)))';
        X1(i,:)=y;
    end
    
    for i=1:n2
        %     Ey2 = .9 + .8*exp(-.8*x).*sin(2*pi*.9*x); % the true function
        Ey2 =  .5 + .8*exp(-.1*x).*sin(.9*pi*x); % the true function
        Ey(:,2) = Ey2;
        
        %Ey1 = 0.01+ 0.3*exp(-10*x); % the true function
        %Ey=Ey+Ey1;
        sigma = 0.1;
        y = (Ey2 + sigma*randn(size(x)))';
        X2(i,:)=y;
    end
    
    for i=1:n3
        %     Ey3 = 1 + 0.5*exp(-.5*x).*sin(2*pi*x); % the true function
        Ey3 = 1 + 0.5*exp(-x).*sin(-1.2*pi*x); % the true function
        Ey(:,3) = Ey3;
        
        %Ey1 = 0.01+ 0.3*exp(-10*x); % the true function
        %Ey=Ey+Ey1;
        sigma = 0.1;
        y = (Ey3 + sigma*randn(size(x)))';
        X3(i,:)=y;
    end
    X =[X1;X2;X3];
end

%% satellite data
if strcmp(data,'satellite')
    satdata = importdata('npfda-sat.dat'); data = satdata;
    X = data;
    klas = [];
end


%% yeast cell cycle data
%set 1

if strcmp(data,'yeast-cellcycle')
    load norm_cellcycle_384_17; 
    klas=Y(:,1);
    X=Y(:,2:end); 
end


% %set 2
% load normcho_237_4class;klas=Y(:,1);Y=Y(:,2:end);
% dataname = 'yeast-cellcycle-mips';


