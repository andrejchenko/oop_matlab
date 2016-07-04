function stacking_svm_Distances_Sunsal_Sum1_5Pix_0NN_pines
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
        cObj.combineMethods_SVM_Distances_SunalSum1(obj,nObj,svmObj,sunObj); % no validation set
   
        cObj.trainData = [sunObj.alphas' svmObj.dist_values ]; % (80 sum to 1 normalized abundances + 120 distances) = 200 feature vector
        cObj.trainLabels = obj.trainLabels;

        %Train meta svm classifier
        metaSvmObj = SVM;
        metaSvmObj.trainSvmMetaUsingExtendedSet(cObj,obj,svmObj,sunObj); % this second level SVM model can output probabilities
        acc = metaSvmObj.svmMetaTestOnDistancesWithExtendedSet(obj,sunObj,cObj);
        avgAccVec = [avgAccVec; acc];
    end
    avgAcc = mean(avgAccVec);
    save avgAcc_Distances avgAcc
    save avgAccVec_Distances avgAccVec
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