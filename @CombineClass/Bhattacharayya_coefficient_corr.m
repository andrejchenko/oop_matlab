function k = Bhattacharayya_coefficient_corr(cObj,prob,alphas)
sumK = 0;
for i = 1:size(prob,1)
    p = prob(i,:);   % 1x 16
    p = p';          % 16 x 1
    
    q = alphas(i,:); % 1x 16
    q = q';          % 16 x 1
    
    
    sumK = sumK + sqrt(p'*q);
end
     k = sumK;
end