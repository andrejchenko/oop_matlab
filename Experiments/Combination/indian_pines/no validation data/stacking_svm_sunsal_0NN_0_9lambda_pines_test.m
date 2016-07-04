function stacking_svm_sunsal_0NN_0_9lambda_pines_test

    [obj,nObj,svmObj,sunObj] = setExperimentParameters();

    obj.load_Indian_Pines();
    obj.selectXPixPerClass_IncludeXNeighbours(nObj);
    if(obj.numNeigh > 0)
        obj.assembleXTrainData(nObj);
    end
    
    cObj = CombineClass;
    %cObj.baseClassification(obj,nObj,svmObj,sunObj);
    %cObj.combineMaxClassifVal(obj,nObj,svmObj,sunObj);
    %cObj.combineMaxClassifValNormedAlphas(obj,nObj,svmObj,sunObj);
    
    %cObj.trainData = [sunObj.alphas' svmObj.prob_values ]; %cObj.trainData is the training set with the extended features 96 features: 16 from svm, 80 from abundances
    cObj.trainData = obj.trainData;
    cObj.trainLabels = obj.trainLabels;
    stop = 1;
    
    %Train meta svm classifier
    metaSvmObj = SVM;
    metaSvmObj.trainSvmMetaClassifier(cObj);
    
    metaSvmObj.svmMetaTest(obj); %obj contains the testData and testLabels
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
end