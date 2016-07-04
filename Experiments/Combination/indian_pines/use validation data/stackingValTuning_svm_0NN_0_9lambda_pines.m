function stackingValTuning_svm_0NN_0_9lambda_pines

    [obj,nObj,svmObj,sunObj] = setExperimentParameters();

    obj.load_Indian_Pines();
    obj.selectXPixPerClass_IncludeXNeighbours(nObj);
    obj.extractValidationData();
    if(obj.numNeigh > 0)
        obj.assembleXTrainData(nObj);
    end
    cObj = CombineClass;
    %cObj.combineMethodsWithValidationData(obj,nObj,svmObj,sunObj);
    %cObj.svmWithValidationData(obj,nObj,svmObj,sunObj);
    svmObj.svmTrainUsingTrainDataTuneUsingValData(obj);
    
    %cObj.trainData = [sunObj.alphas' svmObj.prob_values ]; %cObj.trainData is the training set with the extended features 96 features: 16 from svm, 80 from abundances
    %cObj.trainData = svmObj.prob_values;
    %cObj.trainLabels = obj.trainLabels;
    
    cObj.trainData = svmObj.prob_values_validation;
    cObj.trainLabels = obj.valLabels;
    
    
    %cObj.trainLabels = obj.trainLabels;
    stop = 1;
    
    %Train meta svm classifier
    metaSvmObj = SVM;
    %metaSvmObj.trainSvmMetaClassifier(cObj);
    %metaSvmObj.trainSvmMetaClassifier(cObj,obj);
    metaSvmObj.trainSvmMetaValDataForTuning(cObj,obj,svmObj);
    %metaSvmObj.svmMetaTest(obj); %obj contains the testData and testLabels
    metaSvmObj.svmMetaTestOnProb(obj);
   
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