function [klas, params, Posterior, gmm_density, stored_K, stored_J] = robust_em_RE_PRM(x, Y, p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Robust EM algorithm for Random Effects Polynomial Regression Mixture Model
%
%
%
%
%
% by Faicel Chamroukhi, December 2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% warning off all

[n, m] = size(Y);
Y_in  = Y;

% Construction of the desing matrix
X = designmatrix_Poly_Reg(x,p);%[m x (p+1)]


%n regularly sampled curves
Xstack = repmat(X,n,1);% desing matrix [(n*m) x (p+1)]


Epsilon = 1e-6;

%------ Step 1 initialization ----- %
Beta    = 1;
K       = min(n,1000);
% -----------(28) ---------------- %
gama        = 1e-6;
Y_stack_tmp = repmat(Y',n,[]); % [mxn x n]
Y_stack     = (reshape((Y_stack_tmp(:))',m,[]))'; %[nxn x m];
dij         = sum((Y_stack - repmat(Y,n,[])).^2, 2);
dmin        = min(dij(dij>0));
Q           = dmin;

%%%%
Ytild = reshape(Y',[],1); % []
%%%

%Initialize the mixing proportins
Alphak = 1/K*ones(K,1);
Pik = 1/K*ones(K,1);

% Initialize the regression parameters and the variances
Betak   = zeros(p+1,K);
%d = p+1;
Sigmak2  = zeros(K,1);
for k=1:K
    % ------- step 2  (27)  ------- %
    %betak  = inv(Phi'*Phi + 1e-4*eye(p+1))*Phi'*Y_in(k,:)';
    betak  = (X'*X)\(X'*Y_in(k,:)');
    Betak(:,k) = betak;
    muk = X*betak;
    %Dk = sum((reshape(X,n,m) - reshape(muk',n,m)).^2, 2);
    Dk = sum((Y_in - ones(n,1)*muk').^2, 2);
    Dk = sort(Dk);
    Sigmak2(k)=  Dk(ceil(sqrt(K)));%sum(Y_in(k,:)' - muk);%Dk(ceil(sqrt(K)));%.001;%%;%
    % --------------------------- %
end

%--------- Step 3 (4) --------%

% compute the pposterior cluster probabilites (responsibilities) for the
% initial guess of the model parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           %
%       E-Step              %
%                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PikFik = zeros(n, K); 
log_Pik_Fik = zeros(n,K);

% % E-Step of Gaffney
% [N,D] = size(Y);
% [P,K,D] = size(Mu);
%
% Pik = zeros(n,K,D);
% Beta_ikd = zeros(P,n,K,D);
% Vikd = zeros(P,P,n,K,D);
% for k=1:K
%   for d=1:D
%     Rinv = inv(R(:,:,k,d));
%     for i=1:n
%       indx = Seq(i):Seq(i+1)-1;
%       n_i = length(indx);
%       V_y = X(indx,:)*R(:,:,k,d)*X(indx,:)' + Sigma(i,d)*eye(n_i);
%       Pik(i,k,d) = mvnormpdf(Y(indx,d)',X(indx,:)*Mu(:,k,d),V_y);
%
%       A = 1/Sigma(i,d)*X(indx,:)'*X(indx,:) + Rinv;
%       c = 1/Sigma(i,d)*X(indx,:)'*Y(indx,d) + Rinv*Mu(:,k,d);
%       Beta_ikd(:,i,k,d) = A\c;
%
%       Vikd(:,:,i,k,d) = inv(1/Sigma(i,d)*X(indx,:)'*X(indx,:) + Rinv);
%     end
%   end
% end
% Pik = prod(Pik,3);  % scaling problems?
% Pik = Pik .* (ones(n,1)*Alpha');


% RE

D =p+1;
Ksi_k =rand(K,1);
Ki_ik = zeros(n,K);
lambda_ik = zeros(n,K);
% B_ik = zeros(K,D,n);
B_ik = zeros(D,n,K);

X_i = X;
m_i=m;
for k=1:K
    pik = Pik(k);
    betak = Betak(:,k);
    
    sigmak2 = Sigmak2(k);
    
    %if heteroscedasticity==1,sigmak  = param.sigma_k(k)  ;%variance %[1] ou [1 x K]
    %else sigmak=param.sigmak;end
    %
    
    ksi_k = Ksi_k(k);
    
    Muk = reshape(Xstack*betak, m, n)'; %[n*m];
    
    Sigmaki = ksi_k*(X_i*X_i') + sigmak2*eye(m_i);%[m x m]
    
    
    %fik
    
    z =((Y-Muk)/(Sigmaki)).*(Y-Muk);
    mahalanobis = sum(z,2);
    
    log_fk_Xi = - (m_i/2)*log(2*pi) - 0.5*logdet(Sigmaki) - 0.5*mahalanobis;
    log_Pik_Fik(:,k) = log(pik) + log_fk_Xi;% [n x K]
    
    %
    Ki_ik(:,k)=ones(n,1)*trace(ksi_k*(eye(p+1)-ksi_k*X_i'/(Sigmaki)*X_i));%repmat sur i
    lambda_ik(:,k) = ones(n,1)*trace(ksi_k*X_i*(eye(p+1)-ksi_k*X_i'/Sigmaki*X_i)*X_i');%repmat sur i)
    B_ik(:,:,k) = ksi_k*(X_i'/Sigmaki)*(Y-Muk)'; %[Dxn xK]
end

%Posterior = PikFik./(sum(PikFik,2)*ones(1,K));
log_Prosterior  = log_normalize(log_Pik_Fik);
Posterior       = exp(log_Prosterior);
Tauik           = Posterior;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           %
% main Robust EM-MxReg loop %
%                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stored_J  = []; % to store the maximized penalized log-likelihood criterion
pen_loglik_old = -inf;
iter      = 1; % iteration number
MaxIter   = 1000;
converged = 0;
stored_K  = []; % to store the estimatde number of clusters at each iteration

%RE

wXbk = zeros(n*m,1);
Lambda_ik_tild= zeros(n*m,1);
while(iter<=MaxIter && ~converged)
    stored_K = [stored_K K];
    
    %     % print the value of the optimized criterion
    %     %pen_loglik = sum(log(sum(PikFik,2)),1 ) + Beta*n*sum(Alphak.*log(Alphak));
    %     %pen_loglik = (sum(log_Prosterior(:) .* Posterior(:)) - sum(log_Prosterior(:) .* log_Pik_Fik(:)))+ Beta*n*sum(Alphak.*log(Alphak));
    pen_loglik = sum(logsumexp(log_Pik_Fik,2),1) + Beta*n*sum(Alphak.*log(Alphak));
    fprintf(1,'EM Iteration : %d  | number of clusters K : %d | penalized loglikelihood: %f \n',iter-1, K, pen_loglik);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                           %
    %       M-Step              %
    %                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=1:K
        tauik = Tauik(:,k);
        % ------Step 4 (25) ----------%
        % update of the regression coefficients
        temp =  repmat(tauik,1,m);% [m x n]
        Wk = reshape(temp',[],1);%cluster_weights(:)% [mn x 1]
        % meme chose
        % temp =  repmat(tauik,1,m)';% [m x n]
        % cluster_weights = cluster_weights(:);
        wYk = sqrt(Wk).*Ytild; % fuzzy cluster k
        wXk = sqrt(Wk*ones(1,p+1)).*Xstack;%[(n*m)*(p+1)]
        %% RE 
        bk = B_ik(:,:,k); 
        %         wXbk = sqrt(Wk*ones(1,p+1)).*(Xstack*bk');%[(n*m)*1]
        for i=1:n
            wXbk((i-1)*m+1:i*m) = (sqrt(Wk((i-1)*m+1:i*m)*ones(1,p+1)).*Xstack((i-1)*m+1:i*m,:))*bk(:,i);%[(n*m)*1]
        end
        betak  = (wXk'*wXk + 1e-4*eye(p+1))\(wXk'*(wYk - wXbk));
        Betak(:,k) = betak;
        
        %update the kxi_k's
        %Ksi_k(k)= sum (tauik.*(sum(bk.*bk,1)' + Ki_ik(:,k)))/(D*sum(Wk));
        
        
        %%
%         beta_k = repmat(betak,1,n); % ?
%          Y_c = Y'- X_i*(beta_k + bk);
%        S_Sigma(k) = tauik'*(sum(Y_c.*Y_c)'+lambda_ik(:,k));
        
        %sigmak2
%         lambda_ik_tild = repmat(lambda_ik(:,k)',m,[]);
%         Lambda_ik_tild(:,k) = lambda_ik_tild(:);
%         sigmak2 = sum((wYk - wXk*betak - wXbk).^2 + Lambda_ik_tild(:,k))/sum(Wk);
%         Sigmak2(k) = sigmak2;
        %% 
        % ------ Cooected with step 5 (13) ----------%
        % mixing proportions : alphak_EM
        pik = sum(tauik)/n;%alpha_k^EM
        Pik(k) = pik;
    end
    %RE
%     Sigmak2 = sum(S_Sigma)/(n*m) * ones(K,1);
    %Sigmak2 = (1-gama)*Sigmak2 + gama*Q;
    
    % ------- step 5 (13) ------- %
    AlphakOld = Alphak;
    
    Alphak = Pik + Beta * Alphak.*(log(Alphak)-sum(Alphak .* log(Alphak)));
    
    % ------- step 6 (24)  ------- %
    % update beta
    E = sum(sum(AlphakOld .* log(AlphakOld)));
    eta = min ( 1 , 0.5^floor((m/2) - 1) );
    pik_max = max(Pik); alphak_max = max(AlphakOld);
    Beta = min( sum( exp(-eta*n * abs(Alphak - AlphakOld) ) )/K  , (1 - pik_max) / (-alphak_max * E ) );
    
    % ------- step 7 --------- %
    %Kold = K;
    %update the number of clusters K
    small_klas = find(Alphak < 1/n);
    % ------- step 7  (14) ------- %
    K = K - length(small_klas);

    % discard the small clusters
    Pik(small_klas)              = [];
    Alphak(small_klas)           = [];
%     log_fk_xij(:, small_klas)    = [];
%     log_Pik_fk_Xi(:, small_klas) = [];
    log_Pik_Fik(:,small_klas)    = [];
    PikFik(:,small_klas)         = [];
    log_Prosterior(:, small_klas)= [];
    Posterior(:,small_klas)      = [];
    Sigmak2(small_klas)          = [];
    Betak(:,small_klas)          = [];
    %% RE-PRM part
    Ksi_k(small_klas)            = [];
    Ki_ik(:,small_klas)          = [];
    lambda_ik(:,small_klas)      = [];
%     Lambda_ik_tild(:,small_klas)      = [];
    B_ik(:,:,small_klas)       = [];
    % ------- step 7  (15) normalize the Pik and Alphak ------- %
    Pik     = Pik / sum(Pik);
    Alphak  = Alphak / sum(Alphak);
    % ------- step 7 (16)  normalize the posterior prob ------- %
    Posterior = Posterior./(sum(Posterior,2)*ones(1,K));
    Tauik = Posterior;
    
    % -------- step 7 ------------ %
    % test if the partition is stable (K not changing)
    nit = 60;
    if (iter >= nit) && (stored_K(iter-(nit-1)) - K == 0); Beta = 0;  end
    % -----------step 8 (26) and (28) ---------------- %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                           %
    %       M-Step              %
    %                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=1:K
        tauik = Tauik(:,k);
        %RE-PRM
        temp =  repmat(tauik,1,m);% [m x n]
        Wk = reshape(temp',[],1);
        wYk = sqrt(Wk).*Ytild;
        wXk = sqrt(Wk*ones(1,p+1)).*Xstack;
        %
        bk = B_ik(:,:,k);
        for i=1:n
            wXbk((i-1)*m+1:i*m) = (sqrt(Wk((i-1)*m+1:i*m)*ones(1,p+1)).*Xstack((i-1)*m+1:i*m,:))*bk(:,i);%[(n*m)*1]
        end
        
        betak  = (wXk'*wXk + 1e-4*eye(p+1))\(wXk'*(wYk - wXbk));
        Betak(:,k) = betak;
        
        %update the kxi_k's
        Ksi_k(k)= sum (tauik.*(sum(bk.*bk,1)' + Ki_ik(:,k)))/(D*sum(Wk));
        %%
%         betak = Betak(:,k);
%         
%              %%
        beta_k = repmat(betak,1,n); % ?
         Y_c = Y'- X_i*(beta_k + bk);
       S_Sigma(k) = tauik'*(sum(Y_c.*Y_c)'+lambda_ik(:,k));
    
%         % ----------- (26) ---------------- %
%         % update the variance
%         sigmak2 = sum((wYk - wXk*betak - wXbk).^2 + Lambda_ik_tild(:,k))/sum(Wk);
%         Sigmak2(k) = sigmak2;
%         %         sigmak2 = sum((wYk - wXk*betak).^2)/sum(Wk);
%         % -----------(28) ---------------- %
%         sigmak2 = (1-gama)*sigmak2 + gama*Q;
%         %
%         Sigmak2(k) = sigmak2;
    end
    Sigmak2 = sum(S_Sigma)/(n*m) * ones(K,1);
%     Sigmak2 = (1-gama)*Sigmak2 + gama*Q;

    
    % -----------step 9 (4) ---------------- %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                           %
    %       E-Step              %
    %                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=1:K
        alphak  = Alphak(k);
        betak   = Betak(:,k);
        sigmak2 = Sigmak2(k);
        %
        ksi_k = Ksi_k(k);
        
        Muk = reshape(Xstack*betak, m, n)'; %[n*m];
        
        Sigmaki = ksi_k*(X_i*X_i') + sigmak2*eye(m_i);%[m x m]
        %
        z =((Y-Muk)/(Sigmaki)).*(Y-Muk);
        mahalanobis = sum(z,2);
        log_fk_Xi = - (m_i/2)*log(2*pi) - 0.5*logdet(Sigmaki) - 0.5*mahalanobis;
        
        log_Pik_Fik(:,k) = log(alphak) + log_fk_Xi;% [n x K]
        
        %
        Ki_ik(:,k)=ones(n,1)*trace(ksi_k*(eye(p+1)-ksi_k*X_i'/(Sigmaki)*X_i));%repmat sur i
        lambda_ik(:,k) = ones(n,1)*trace(ksi_k*X_i*(eye(p+1)-ksi_k*X_i'/Sigmaki*X_i)*X_i');%repmat sur i)
        B_ik(:,:,k) = ksi_k*(X_i'/Sigmaki)*(Y-Muk)'; %[K x D x n]
    end
    % PikFik = exp(log_Pik_Fik);
    %Posterior = PikFik./(sum(PikFik,2)*ones(1,K));
    %Posterior = exp(log_normalize(log_Pik_fk_Xi));
    log_Posterior = log_normalize(log_Pik_Fik);
    Posterior = exp(log_normalize(log_Posterior));
    Tauik = Posterior;
    
    %%%%%%%%%%
    % compute the value of the optimized criterion J (12) %
    %pen_loglik = sum(log(sum(PikFik,2)),1 ) + Beta*n*sum(Alphak.*log(Alphak));
    %pen_loglik = sum(logsumexp(log_Pik_Fik,2),1) + Beta*n*sum(Alphak.*log(Alphak));
    %pen_loglik = (sum(log_Prosterior(:) .* Posterior(:)) - sum(log_Prosterior(:) .* log_Pik_Fik(:)))+ Beta*n*sum(Alphak.*log(Alphak));
    stored_J = [stored_J pen_loglik];
    %     fprintf(1,'EM Iteration : %d  | number of clusters K : %d | penalized loglikelihood: %f \n',iter, K, pen_loglik);
    %%%%%%%%%
    
    % -----------step 10 (25) ---------------- %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                           %
    %       M-Step              %
    %                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    BetakOld = Betak;
    for k=1:K
        tauik =Tauik(:,k);
        
        % update of the regression coefficients
        %RE-PRM
        temp =  repmat(tauik,1,m);% [m x n]
        Wk = reshape(temp',[],1);
        wYk = sqrt(Wk).*Ytild;
        wXk = sqrt(Wk*ones(1,p+1)).*Xstack;
        %
        bk = B_ik(:,:,k);
        for i=1:n
            wXbk((i-1)*m+1:i*m) = (sqrt(Wk((i-1)*m+1:i*m)*ones(1,p+1)).*Xstack((i-1)*m+1:i*m,:))*bk(:,i);%[(n*m)*1]
        end
        betak  = (wXk'*wXk + 1e-4*eye(p+1))\(wXk'*(wYk - wXbk));
        %Betak(:,k) = betak;
        %update the kxi_k's
        Ksi_k(k)= sum (tauik.*(sum(bk.*bk,1)' + Ki_ik(:,k)))/(D*sum(Wk));
        
        %beta_k = repmat(betak,1,n);
        %%%
        Betak(:,k) = betak;
    end
    % -----------step 11 ---------------- %
    % test of convergence
    
    distBetak = sqrt(sum((Betak - BetakOld).^2, 2));
    if (max(distBetak) < Epsilon || abs((pen_loglik - pen_loglik_old)/pen_loglik_old)<Epsilon);
        converged = 1;
    end
    pen_loglik_old = pen_loglik;
    
    iter=iter+1;
    
end% en of the Robust EM loop

[~, klas] = max (Posterior,[],2);

gmm_density   = sum(PikFik,2);
params.Pik    = Pik;
params.Alphak = Alphak;
params.Betak  = Betak;
params.Muk    = X*Betak;
params.Sigmak2 = Sigmak2;
end



