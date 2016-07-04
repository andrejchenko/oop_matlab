function stacking_WithValSetForTrain_ARD_svm_sunsal_sum1_sum_0NN_pines
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
        cObj.trainLabels = obj.valLabels;
        
        type = 'classification';
        L_fold = 10;
        method = 'continious';
        
        model = initlssvm(cObj.trainData,cObj.trainLabels,type,[],[],'RBF_kernel');
        model = tunelssvm(model,'simplex','crossvalidatelssvm',{L_fold,'misclass'});
        %model = trainlssvm(model);
        
        [dimensions, ordered, costs, sig2s, model] = bay_lssvmARD(model, method, type);
        
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