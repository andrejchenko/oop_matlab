function sunsal_without_positivity_pavia()
    iter = 100;
    avgAccVec = zeros(iter,1);
    
    for a = 1:iter
        [obj,nObj,sObj] = setExperimentParameters();
        obj.load_Pavia();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        sObj.unmixing_positivity_no(obj);
        acc = obj.acc;
        avgAccVec(a) = obj.acc;
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