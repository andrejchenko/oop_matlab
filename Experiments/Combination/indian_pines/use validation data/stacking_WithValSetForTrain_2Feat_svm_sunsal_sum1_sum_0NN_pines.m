function stacking_WithValSetForTrain_2Feat_svm_sunsal_sum1_sum_0NN_pines
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

        %TO DO:
        %Find the max posterior prob and max abunance value and use them as 2 features to create the new dataset
        
        %cObj.trainData = [sunObj.alphas' svmObj.prob_values ]; % they are coming from the validation set - set 2
        cObj.trainLabels = obj.valLabels;
        cObj.trainData = [max(sunObj.alphas)' max(svmObj.prob_values')' cObj.trainLabels];

        %Train the SVM meta classifier - model using the validation data
        metaSvmObj = SVM;
        metaSvmObj.trainSvmMetaClassifier(cObj,obj); %use this model for the testing phase
        %later assign E = obj.trainData; containing as endmembers initial
        %train data + validation data

        acc = metaSvmObj.svmMetaTestOnFeatureTestSetUsingNewModelSum(obj,sunObj,cObj);
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