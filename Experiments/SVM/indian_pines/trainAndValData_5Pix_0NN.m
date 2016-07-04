function trainAndValData_5Pix_0NN
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
        %Merge train and validation data
        obj.trainData = [obj.trainData; obj.valData];
        obj.trainLabels = [obj.trainLabels; obj.valLabels];
        
        acc = svmClassificationAcc(svmObj,obj);
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