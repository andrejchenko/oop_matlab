function [ betas ] = multinomialLogisticRegressionL1( trainData, trainLabels, rho)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
nclass = max(trainLabels(:));
nfeat = size(trainData, 2);
nParam = (nclass-1)*nfeat; % total number of beta parameters
n = size(trainData, 1);

X = trainData;
Y = trainLabels;

betaCurr = zeros(nclass-1,nfeat);
funLast = 1e+9;
resid = funLast;

iterNum = 1;
maxIter = 10000;
residConvThr = 1e-3;
%stepLen = 1e-1;
stepLen = 0.3;
probs = zeros(nclass,n);

Ind = zeros(nclass-1,n);

for i=1:nclass-1
    Ind(i,:) = (Y==i);
end

while iterNum < maxIter && resid > residConvThr
    
    probs = calculateLogisticRegressionProbs(X, betaCurr);

    Q0 = 0.0;
    for j=1:n
        Q0 = Q0 - log(probs(Y(j),j));
    end
    
    betaFlat = reshape(betaCurr, [1,nParam]);
    gBeta = Q0 + (rho/2)*(betaFlat*betaFlat');
    
    fprintf('Iteration %d, obj. func. %f, neg.log.lik. %f\n', iterNum, gBeta, Q0);
    
    grad = (probs(1:nclass-1,:)-Ind)*X;
    
    betaCurr = betaCurr - stepLen * grad;
    
    iterNum = iterNum + 1;
    resid = abs(funLast - gBeta);
    funLast = gBeta;
end

betas = betaCurr;

numCorrect = 0;
for i=1:n
    [maxProb,ind] = max(probs(:,i));
    if ind==Y(i)
        numCorrect = numCorrect + 1;
    end
end

fprintf('Overall classification accuracy: %d/%d=%f\n', numCorrect, n, numCorrect/n);
