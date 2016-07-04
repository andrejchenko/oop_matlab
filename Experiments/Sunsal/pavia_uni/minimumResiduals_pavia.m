function minimumResiduals_pavia()
    iter = 100;
    avgAccVec = zeros(iter,1);
    
    for a = 1:iter
        [obj,nObj,sObj] = setExperimentParameters();
        obj.load_Pavia();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        sObj.unmixing(obj);
        E = obj.trainData';
        obj.r_perClass = zeros(size(obj.testData,1),obj.numClasses); % N x numClasses
        labels = sObj.est_y_per_class(obj,E);
        
        EVAL_APHA = obj.calcXAccuracy(obj.testLabels,labels);
        acc = EVAL_APHA(1)*100;
        avgAccVec(a) = acc;
        save avgAccVec avgAccVec
    end
    avgAcc = mean(avgAccVec);
    save avgAcc avgAcc
    save avgAccVec avgAccVec
end

function [obj,nObj,sObj] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    sObj = Sunsal;
    % Parameters
    obj.numPix = 5;
    obj.numNeigh = 0;
    obj.numClasses = 9;
    obj.lambda = 0.9;
end