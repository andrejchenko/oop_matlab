function [ ] = evaluateLogisticRegressionModel( testData, testLabels, beta )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
theProbs = calculateLogisticRegressionProbs(testData, beta);
n = size(testData,1);

nclass = size(beta,1) + 1;
confMatrix = zeros(nclass,nclass);

for i=1:n
    [maxProb,ind] = max(theProbs(:,i));
    confMatrix(ind,testLabels(i)) = confMatrix(ind,testLabels(i)) + 1;
end

ovAcc = sum(diag(confMatrix)) / n;

fprintf('Overall accuracy on test data: %f, confusion matrix: \n', ovAcc);

disp(confMatrix);

end
