function trainAndValData_Pix5_0NN_lambda_0_9()
    avgUnmixingVec = [];
    iter = 50;
    for a = 1:iter
        [obj,nObj,sObj] = setExperimentParameters();
        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        obj.extractValidationData();
        %Merge train and validation data
        obj.trainData = [obj.trainData; obj.valData];
        obj.trainLabels = [obj.trainLabels; obj.valLabels];

        acc = sObj.unmixing(obj);
        %acc = obj.acc;
        avgUnmixingVec = [avgUnmixingVec; acc]; 
        
    end
    avgUnmixing = mean(avgUnmixingVec);
end

function [obj,nObj,sObj] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    sObj = Sunsal;
    % Parameters
    obj.numPix = 5;
    obj.numNeigh = 0;
    obj.numClasses = 16;
    obj.lambda = 0.9;
    obj.numValPixPerClass = 5;
end