function TrainPixels10_forLearning_KSVD_dictionary()
avgAccVec = [];
    iter = 100;
    for a = 1:iter
         [obj,nObj,svmObj,sunObj,param] = setExperimentParameters();
%         
        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        
        D = obj.trainData'; % should be d x N 
        %=============================================
        % Run the NN-KSVD function
        %=============================================
        [Dictionary,output] = KSVD_NN(D,param);
        % We will only use the learned dictionary for the unmixing method
        
%         obj.trainData = trainData;
%         obj.trainLabels = trainLabels;
%         obj.testData = testData;
%         obj.testLabels = testLabels;

        obj.trainData = (Dictionary*output.CoefMatrix)';
        %obj.trainData = Dictionary';
        
        for i = 0:0.1:1
            obj.lambda = i;
            sunObj.unmixing(obj);
            acc = obj.acc;
        end

        avgAccVec = [avgAccVec; acc];
    end
    avgAcc = mean(avgAccVec);
    save avgAcc_svmDistancesSum avgAcc
    save avgAccVec_svmDistancesSum avgAccVec
end

function [obj,nObj,svmObj,sunObj,param] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    svmObj = SVM;
    sunObj = Sunsal;
    % Parameters
    obj.numPix = 10;
    obj.numNeigh = 0;
    obj.numClasses = 16;
    % obj.lambda = 0.9;
    param.K = 80; % We want 80 endmembers in the learned dictionary
    param.L = 5;  % We want max 5 abundances to be used in the Orthogonal Matching Pursuit method used for the sparse coding stage. 
    param.InitializationMethod = 'DataElements';
    param.numIteration = 200;
    param.displayProgress = 0;
    param.preserveDCAtom = 0;
end