function plot_results_robust_em_RM(X, K_hat, klas_hat, params, posterior_prob, gmm, stored_J, stored_K)

% plot the results of a robust EM for Regression Mixtures (polynomial,
% Spline, B-Spline, RM with mixed random effects)

[n,m]=size(X);
% x=linspace(0,1,m);
x=1:m;

Ey = params.Muk;

% original data

figure, plot(x,X','r-','linewidth',0.001);    
ylabel('y') 
xlabel('x') 
xlim([min(x) max(x)]);
%set(gca,'ytick',[0.4:0.2:1.4])
box on;
% title(['Robust EM-MixReg clustering : iteration ', int2str(length(stored_J)), '; K = ', int2str(K_hat)]); 
title(['Original data']); 
%
%

%


colors = {'r','b','g','m','c','k','y','r','b','g','m','c','k','y'};

for k=1:K_hat%min(K_hat,7)
    figure
    
    sigmak2 = sqrt(params.Sigmak2(k));
    Ic_k = [Ey(:,k)-2*sigmak2 Ey(:,k)+2*sigmak2];
        
    
    hold on,
    plot(x,X(klas_hat==k,:)','r-','linewidth',0.001);    
    hold on
	plot(x,params.Muk(:,k),'k-','linewidth',5);
    
    hold on
        plot(x,Ic_k,'k--','linewidth',1);

        
        
    hold on
   
    ylabel('y')
xlabel('x') 
xlim([min(x) max(x)]);
title(['Robust EM-MixReg clustering : iteration ', int2str(length(stored_J)), '; K = ', int2str(K_hat)]); 
    box on
end
% ylabel('y')
% xlabel('x') 
% xlim([min(x) max(x)]);
%set(gca,'xtick',[0:0.2:1])
%set(gca,'ytick',[0.4:0.2:1.4])
box on;
% title(['Robust EM-MixReg clustering : iteration ', int2str(length(stored_J)), '; K = ', int2str(K_hat)]); 
%


%%%%%%%
figure,
for k=1:K_hat%min(K_hat,7)
    sigmak2 = sqrt(params.Sigmak2(k));
    Ic_k = [Ey(:,k)-2*sigmak2 Ey(:,k)+2*sigmak2];
    hold on,
    %plot(x,X(klas_hat==k,:)','r','linewidth',0.001);    
    %hold on
    plot(x,params.Muk(:,k),colors{k},'linewidth',5);
    %hold on
    %plot(x,Ic_k,[colors{k},'--'],'linewidth',1);
end
ylabel('y')
xlabel('x') 
xlim([min(x) max(x)]);
box on;
title(['Robust EM-MixReg clustering (cluster centers) : iter ', int2str(length(stored_J)), '; K = ', int2str(K_hat)]); 

%%%%%%%
figure,

    if(K_hat==2)
        ha = tight_subplot(1,2,[.01 .01],[.01 .01],[.01 .01]);
    elseif (K_hat==3)
        ha = tight_subplot(1,3,[.01 .1],[.1 .1],[.1 .1]);
    elseif (K_hat==4)
        ha = tight_subplot(2,2,[.01 .01],[.01 .01],[.01 .01]);
    elseif (K_hat==5)
        ha = tight_subplot(3,2,[.01 .01],[.01 .01],[.01 .01]);
    elseif (K_hat==6)
        ha = tight_subplot(3,2,[.01 .01],[.01 .01],[.01 .01]);
    elseif (K_hat==7)
        ha = tight_subplot(4,2,[.01 .01],[.01 .01],[.01 .01]);
    elseif (K_hat==8)
        ha = tight_subplot(4,2,[.01 .01],[.01 .01],[.01 .01]);
    end
title(['Robust EM-MixReg clustering : iteration ', int2str(length(stored_J)), '; K = ', int2str(K_hat)]); 
ylabel('y')
    
for k=1:K_hat%min(K_hat,7)
    sigmak2 = sqrt(params.Sigmak2(k));
    Ic_k = [Ey(:,k)-2*sigmak2 Ey(:,k)+2*sigmak2];
    %if mod(K,2), subplot(K/3,2,k);else subplot(K/2,2,k);end
%     if(K_hat==2)
%         subplot(1,2,k);
%     elseif (K_hat==3)
%         subplot(3,1,k);
%     elseif (K_hat==4)
%         subplot(2,2,k);
%     elseif (K_hat==5)
%         subplot(3,2,k);
%     elseif (K_hat==6)
%         subplot(3,2,k);
%     elseif (K_hat==7)
%         subplot(4,2,k);
%     elseif (K_hat==8)
%         subplot(4,2,k);
%     end
    axes(ha(k));
    plot(x,X(klas_hat==k,:)','r','linewidth',0.001);    
    hold on
    plot(x,params.Muk(:,k),'k','linewidth',5);
    hold on
    plot(x,Ic_k,'k--','linewidth',1);
    xlabel('x') 
end
xlim([min(x) max(x)]);
box on;


%%%%%%%
figure
plot(stored_J,'b-x');
xlabel('Robust EM-MixReg iteration number');
ylabel('penalized observed data log-likelihood');box on;

%%%%%%


%%%%%%%
figure
semilogy(stored_K,'b->','markersize',10);
xlabel('Robust EM-MixReg iteration number');
ylabel('Number of clusters K');
ylim([2 max(stored_K)])
set(gca,'ytick',unique(stored_K))
box on;


