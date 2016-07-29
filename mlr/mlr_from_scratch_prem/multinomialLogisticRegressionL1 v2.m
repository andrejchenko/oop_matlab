function [ betas ] = multinomialLogisticRegressionL1( trainData, trainLabels, lambda, alpha, verbose)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin < 5
    verbose = 0;
end

nclass = max(trainLabels(:));
nfeat = size(trainData, 2);
nParam = (nclass-1)*nfeat;
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

%XXX
%betaCurr = 100000*ones(nclass-1,nfeat);

%regBalanceCoeff = n/nParam;
regBalanceCoeff = 1.0;

proxOp = @(x) softThreshold(x,stepLen*lambda*alpha);
rho = lambda*(1-alpha);

while iterNum < maxIter && resid > residConvThr
    
    probs = calculateLogisticRegressionProbs(X, betaCurr);

    Q0 = 0.0;
    for j=1:n
        Q0 = Q0 - log(probs(Y(j),j));
    end
    
    betaFlat = reshape(betaCurr, [1,nParam]);
    regTerm = (rho/2)*(betaFlat*betaFlat');
    gBeta = Q0 + regBalanceCoeff*regTerm;
    
    hBeta = lambda*alpha*norm(betaCurr,1);
    
    Q1 = gBeta + hBeta;
    
    %XXX testing
    %gBeta = regTerm;
    
    if verbose > 0
        fprintf('Iteration %d, obj. func. %f, neg.log.lik. %f\n', iterNum, gBeta, Q0);
    end
    
    gradReg = rho*betaCurr;
    gradQ0 = (probs(1:nclass-1,:)-Ind)*X;
    grad = gradQ0 + regBalanceCoeff*gradReg;
    
    %XXX testing
    %grad = gradReg;
    
    betaCurr = proxOp(betaCurr - stepLen * grad);
    
    %betaCurr = betaCurr - stepLen * grad;
    
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

if verbose > 0
    fprintf('Overall classification accuracy (train): %d/%d=%f\n', numCorrect, n, numCorrect/n);
end
