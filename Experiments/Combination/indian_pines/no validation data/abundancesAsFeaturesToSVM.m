function abundancesAsFeaturesToSVM()
avgAccVec = [];
    iter = 100;
    for a = 1:iter
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();
        cObj = CombineClass;

        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        trainData = obj.trainData;
        
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        %Extract abundances as feature vectors
        sunObj.unmixingTrainData(obj); 
        %Now we will use the abundances - sunObj.alphas as input features to the SVM
        cObj.sumToOneNorm(sunObj);
        cObj.sumAlphasPerClassForTrainData(obj,sunObj);
        obj.trainData = sunObj.alphas'; % to have N x p trainData
        svmObj.svmTrainOnly(obj);
       
        %Convert test data to abundances
        obj.trainData = trainData;
        sunObj.unmixing(obj); %sunObj.alphas are now the test data
        cObj.sumToOneNorm(sunObj);
        cObj.sumAlphasPerClassForTestData(obj,sunObj);
        obj.testData = sunObj.alphas'; % to have N x p testData
        svmObj.svmClassifyOnly(obj);
        acc = svmObj.accuracy(1);
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