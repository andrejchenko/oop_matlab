function ls_svm_5Pix()

avgAccVec = [];
iter = 100;
for i = 1:iter
    type = 'classification';
    L_fold = 10;

    [obj,nObj,svmObj,sunObj] = setExperimentParameters();

    obj.load_Indian_Pines();
    obj.selectXPixPerClass_IncludeXNeighbours(nObj);

    model = initlssvm(obj.trainData,obj.trainLabels,type,[],[],'RBF_kernel');
    %pitchModel = initlssvm(reducedPitchImgs,targetPitchDegree,type,[],[],'lin_kernel');
    model = tunelssvm(model,'simplex','crossvalidatelssvm',{L_fold,'misclass'});
    % pitchModel = tunelssvm(pitchModel,'simplex','leaveoneoutlssvm',{'mse'});
    model = trainlssvm(model);

    % Validate the class model on the test images
    %output = simlssvm(model,obj.testData);
    [output, Zt] = simlssvm(model,obj.testData);

    acc = obj.calcXAccuracy(obj.testLabels,output);
    acc = acc(1)*100;
    avgAccVec = [avgAccVec; acc];
    stop = 1;
end
avgAcc = mean(avgAccVec);
save avgAcc_ls_svm avgAcc
save avgAccVec_svm avgAccVec
stop = 1;
end

function [obj,nObj,svmObj,sunObj] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    svmObj = SVM;
    sunObj = Sunsal;
    % Parameters
    obj.numPix = 5;
    obj.numNeigh = 0;
    obj.numClasses = 16;
    obj.lambda = 0.9;
    obj.numValPixPerClass = 5;
end
