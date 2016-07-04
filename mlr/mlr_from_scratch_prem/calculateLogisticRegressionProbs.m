function [ probs ] = calculateLogisticRegressionProbs( X, beta )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    n = size(X,1);
    nclass = size(beta,1)+1;
    probs = zeros(nclass,n);
    for j=1:n
        ps = zeros(nclass-1,1);
        for i=1:nclass-1
            ps(i) = exp(beta(i,:)*X(j,:)');
        end
        
        sumP = sum(ps(:))+1;
        
        for i=1:nclass-1
            probs(i,j) = ps(i) / sumP;
        end
        
         probs(nclass,j) = 1-sum(probs(1:nclass-1,j));
    end

end
