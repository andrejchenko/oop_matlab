function stacking_svmNormUnit_sunsalNormSum1Max_NoVal_0NN_0_9lam_pines
    avgUnmixing = 0;
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
        cObj.combineMethodsSVMUnitNormedSunalSum1MaxNoValidation(obj,nObj,svmObj,sunObj);

        cObj.trainData = [sunObj.alphas' svmObj.prob_values ];
        cObj.trainLabels = obj.trainLabels;

        %Train meta svm classifier
        metaSvmObj = SVM;
        metaSvmObj.trainSvmMetaUsingExtendedSet(cObj,obj,svmObj,sunObj);
        %acc = metaSvmObj.svmMetaTestOnProbWithExtendedSetAvg(obj,sunObj,cObj);
        acc = metaSvmObj.svmMetaTest_ProbUnitNormed_AbunSum1Max(obj,sunObj,cObj);
        avgUnmixing = avgUnmixing + acc;
        fileName = fullfile('Experiments','Combination','indian_pines','results','text','stacking_svmNormUnit_sunsalNormSum1Max_NoVal_0NN_0_9lam.txt');
        fileID = fopen(fileName,'a');
        fprintf(fileID,'Iteration: %d, Combined accuracy: %4.3f with fixed lambda: %4.3f\n',a,acc);
        fclose(fileID);
        stop1 = 1;
    end
    avgUnmixing = avgUnmixing/iter;
    fileName = fullfile('Experiments','Combination','indian_pines','results','text','stacking_svmNormUnit_sunsalNormSum1Max_NoVal_0NN_0_9lam.txt');
    fileID = fopen(fileName,'a');
    fprintf(fileID,'Average combined accuracy: %4.3f\n',avgUnmixing);
    fclose(fileID);
    stop2 = 1;
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