function param = initialize_MixReg(data,G , phiBeta, init_kmeans, try_algo) 
%%%%%%%%%%%%%%%%%%%%
%
%
%
%
%
%%%%%%%%%%%%%%%%%% FC
[n m]=size(data);
p = size(phiBeta,2)-1;
 
%Initialization of the model parameters for each cluster: W (pi_jgk), betak and sigmak  
% 1. Initialization of cluster weights
param.Pi_k=1/G*ones(G,1);

% 2. betagk and sigmagk
if init_kmeans
    D = data;       
    max_iter_kmeans = 400;
    n_tries_kmeans = 20;
    verbose_kmeans = 0;
    
    res_kmeans = Kmeans_faicel(D,G,n_tries_kmeans, max_iter_kmeans, verbose_kmeans);
    
    klas = res_kmeans.klas;
    
    for g=1:G
        Xg = D(klas==g ,:); %if kmeans 
        
        clusterg_labels =  repmat(klas,1,m)'; 
        clusterg_labels = clusterg_labels(:);
        phiBetag = phiBeta(clusterg_labels==g,:);

        param_init =  init_regression_param(Xg, phiBetag, try_algo);   
        param.beta_k(:,g) = param_init.beta;         
        param.sigma_k(g) = param_init.sigma;
     end
else
    klas = zeros(n,1);
    ind = randperm(n);
    D=data;
    for g=1:G
        if g<G
            ind_klasg=ind((g-1)*round(n/G) +1 : g*round(n/G));
            klas(ind_klasg)=g;
            Xg = D(ind_klasg,:);
        else%g=G
            ind_klasG = ind((g-1)*round(n/G) +1 : end);
            klas(ind_klasG)=G;
            Xg = D(ind_klasG,:);
        end
        
        clusterg_labels =  repmat(klas,1,m)';
        clusterg_labels = clusterg_labels(:);
        phiBetag = phiBeta(clusterg_labels==g,:);
        param_init =  init_regression_param(Xg, phiBetag, try_algo);
        
        param.beta_k(:,g) = param_init.beta;                 
        param.sigma_k(g) = param_init.sigma;
     end
end

%%%%%%%%%%%%%%%%%%%%%%
function para = init_regression_param(data, phi, try_algo)
 
%  
%
%
%
%%%%%%%%%%%%%%%%%%%% Faicel Chamroukhi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n m] = size(data);
p = size(phi,2)-1;
Y = data;
Y  = reshape(Y',[],1); 

X = phi;

if try_algo == 1        
   beta = inv(X'*X)*X'*Y;
   para.beta = beta;
   para.sigma  = sum((Y - X*beta).^2)/(n*m); 
else % initialisation aléatoire      
    para.beta =  rand(p+1,1);         
    para.sigma = rand(1) ; 
end
