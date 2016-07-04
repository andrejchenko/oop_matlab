function svm_5Pix_0NN()

    [obj,nObj,svmObj] = setExperimentParameters();
    %obj.load_Indian_Pines();
    
    %obj.selectXPixPerClass_IncludeXNeighbours(nObj);
    load('trainData');
    load('trainLabels');
    load('testData');
    load('testLabels');
    
    obj.trainData = trainData;
    obj.trainLabels = trainLabels;
    obj.testData = testData;
    obj.testLabels = testLabels;
    
    svmClassification(svmObj,obj);
    
    str = ['SVM accuracy: ', num2str(svmObj.accuracy(1))];
    str
    svmAccuracy = svmObj.accuracy(1);
end

function [obj,nObj,svmObj] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    svmObj = SVM;
    % Parameters
    obj.numPix = 5;
    obj.numNeigh = 0;
    obj.numClasses = 16;
end