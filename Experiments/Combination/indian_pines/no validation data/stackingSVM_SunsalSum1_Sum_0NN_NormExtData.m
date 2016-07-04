function stackingSVM_SunsalSum1_Sum_0NN_NormExtData
    avgAccVec = [];
    iter = 100;
    for a = 1:iter
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();

        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        %obj.extractValidationData();
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        cObj = CombineClass;
        cObj.combineMethodsSVMSunalNoValidationSum(obj,nObj,svmObj,sunObj);

        cObj.trainData = [sunObj.alphas' svmObj.prob_values ];
        cObj.trainLabels = obj.trainLabels;
        
        %Normalize the extended dataset -> zscore
        cObj.trainData = zscore(cObj.trainData);

        %Train meta svm classifier
        metaSvmObj = SVM;
        metaSvmObj.trainSvmMetaUsingExtendedSet(cObj,obj,svmObj,sunObj);
        acc = metaSvmObj.svmMetaTestOnProbWithExtendedSetSum(obj,sunObj,cObj);
        avgAccVec = [avgAccVec; acc];
    end
    avgAcc = mean(avgAccVec);
    save avgAccVec avgAccVec
    save avgAcc avgAcc
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