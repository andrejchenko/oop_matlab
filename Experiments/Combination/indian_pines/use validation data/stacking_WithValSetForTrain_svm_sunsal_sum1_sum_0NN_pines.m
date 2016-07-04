function stacking_WithValSetForTrain_svm_sunsal_sum1_sum_0NN_pines
 avgUnmixing = [];
    iter = 50;
    for a = 1:iter
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();

        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        obj.extractValidationData();
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        cObj = CombineClass;
        cObj.combineMethodsWithValDataUsedToTrainSum(obj,nObj,svmObj,sunObj);
        %cObj.combineMethodsSVMSunalAvg(obj,nObj,svmObj,sunObj);
        %cObj.svmWithValidationData(obj,nObj,svmObj,sunObj);
        %svmObj.svmTrainUsingTrainDataTuneUsingValData(obj);

        cObj.trainData = [sunObj.alphas' svmObj.prob_values ]; % they are coming from the validation set - set 2
        %cObj.trainData = [sunObj.alphas_validation' svmObj.prob_values_validation ]; %cObj.trainData is the validation set with the extended features 32 features: 16 from svm, 1 from abundances
        %for tunning the parameters of the svm meta classifier
        cObj.trainLabels = obj.valLabels;
        
        %train the SVM model on the whole: train + validation data
        svmTrainUsingTrainAndValidationData(svmObj,obj); %use this model for the testing phase
        %later assign E = obj.trainData; containing as endmembers initial
        %train data + validation data

        %Train meta svm classifier
        metaSvmObj = SVM;
        %metaSvmObj.trainSvmMetaClassifier(cObj);
        %metaSvmObj.trainSvmMetaClassifier(cObj,obj);
        metaSvmObj.trainSvmMetaWithExtendedSetFromValData(cObj,obj,svmObj,sunObj);
        %metaSvmObj.svmMetaTest(obj); %obj contains the testData and testLabels
        acc = metaSvmObj.svmMetaTestOnTestExtendedSetUsingNewModelSum(obj,sunObj,cObj);
        avgUnmixing = [avgUnmixing; acc]; 
    end
    save avgAcc avgUnmixing;
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