function stackingValTuning_svm_sunsal_Max_0NN_0_9lambda_pines
 avgUnmixing = 0;
    iter = 100;
    for a = 1:iter
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();

        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        obj.extractValidationData();
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        cObj = CombineClass;
        %cObj.combineMethodsWithValidationData(obj,nObj,svmObj,sunObj);
        cObj.combineMethodsSVMSunalAvg(obj,nObj,svmObj,sunObj);
        %cObj.svmWithValidationData(obj,nObj,svmObj,sunObj);
        %svmObj.svmTrainUsingTrainDataTuneUsingValData(obj);

        cObj.trainData = [sunObj.alphas_validation' svmObj.prob_values_validation ]; %cObj.trainData is the validation set with the extended features 32 features: 16 from svm, 1 from abundances
        %for tunning the parameters of the svm meta classifier
        cObj.trainLabels = obj.valLabels;

        %Train meta svm classifier
        metaSvmObj = SVM;
        %metaSvmObj.trainSvmMetaClassifier(cObj);
        %metaSvmObj.trainSvmMetaClassifier(cObj,obj);
        metaSvmObj.trainSvmMetaValDataForTuningExtendedSet(cObj,obj,svmObj,sunObj);
        %metaSvmObj.svmMetaTest(obj); %obj contains the testData and testLabels
        acc = metaSvmObj.svmMetaTestOnProbWithExtendedSetAvg(obj,sunObj,cObj);
        
        avgUnmixing = avgUnmixing + acc;
        fileName = fullfile('Experiments','Combination','indian_pines','results','text','stacking_svm_sunsal_Avg_5Pix_0NN_0_9lambda.txt');
        fileID = fopen(fileName,'a');
        fprintf(fileID,'Iteration: %d, Combined accuracy: %4.3f with fixed lambda: %4.3f\n',a,acc);
        fclose(fileID);
        stop1 = 1;
    end
    avgUnmixing = avgUnmixing/iter;
    fileName = fullfile('Experiments','Combination','indian_pines','results','text','stacking_svm_sunsal_Avg_5Pix_0NN_0_9lambda.txt');
    fileID = fopen(fileName,'a');
    fprintf(fileID,'Average combined accuracy: %4.3f \n',avgUnmixing);
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