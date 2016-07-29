rhoVals = logspace(-7,1,9);

ovAccs = zeros(length(rhoVals),1);

for i=1:length(rhoVals)
    theBeta=multinomialLogisticRegressionL1(trainData,trainLabels,rhoVals(i),1);
    ovAccs(i) = evaluateLogisticRegressionModel(testData,testLabels,theBeta);
    fprintf('Ov. accuracy for rho=%f: \n',rhoVals(i),ovAccs(i));
end

[bestAcc,bestIdx] = max(ovAccs);

fprintf('Best ov. accuracy of %f attained for rho=%f\n',bestAcc,rhoVals(bestIdx));
