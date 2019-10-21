function pourcent_misclassified =evaluation(klas_vrai,klas_estim)
    crtb=crosstab(klas_vrai,klas_estim);
    k=length(crtb);
    a=perms([1:k]);
    for i=1:factorial(k)
        for j=1:k
          Y(i,j)=crtb(j,a(i,j));
        end
    end   
    tab=sum(Y,2);
    maxi=max(tab);
    
n=length(klas_vrai);
nbre_mal_classes = n-maxi;
pourcent_misclassified = (nbre_mal_classes/n)*100;% err en %