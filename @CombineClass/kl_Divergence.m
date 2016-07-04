%Base P and Q on the same set of outcomes
%http://stats.stackexchange.com/questions/97938/calculate-the-kullback-leibler-divergence-in-practice
%Normalization of the KL Divergence values:
%http://math.stackexchange.com/questions/51482/can-i-normalize-kl-divergence-to-be-leq-1
function n_kl_divergences = kl_Divergence(cObj,prob,alphas)
    div = [];
    for i = 1: size(prob,1)
        v1 = alphas(i,:); %P(i) = alphas
        v2 = prob(i,:);   %Q(i) = probabilities  

        sumD = 0;
        for j = 1:size(v1,2)
            if(v1(j)~=0)
                d = v1(j)*log(v1(j)/v2(j));
                sumD = sumD + d;
            end
        end
        div = [div; sumD];
    end
    
    kl_divergences = div;
    n_kl_divergences = 1-exp(-kl_divergences); % Normalize KL-divergences in the [0-1] range
    save n_kl_divergences n_kl_divergences
    mean_kl_divergence = mean(n_kl_divergences);
    save mean_kl_divergence mean_kl_divergence
    %save kl_divergences
end