function stacking_svm_Distances_Sunsal_Sum1_Sum_5Pix_0NN_pines
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
        cObj.combineMethods_SVM_Distances_SunalSum1Sum_NoValidation(obj,nObj,svmObj,sunObj);

        cObj.trainData = [sunObj.alphas' svmObj.dist_values ]; % (16 abundances + 120 distances) = 136 feature vector
        cObj.trainLabels = obj.trainLabels;

        %Train meta svm classifier
        metaSvmObj = SVM;
        metaSvmObj.trainSvmMetaUsingExtendedSet(cObj,obj,svmObj,sunObj); % this second level SVM model can output probabilities
        acc = metaSvmObj.svmMetaTestOnDistancesWithExtendedSetSum(obj,sunObj,cObj);
        avgAccVec = [avgAccVec; acc];
    end
    avgAcc = mean(avgAccVec);
    save avgAcc_svmDistancesSum avgAcc
    save avgAccVec_svmDistancesSum avgAccVec
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