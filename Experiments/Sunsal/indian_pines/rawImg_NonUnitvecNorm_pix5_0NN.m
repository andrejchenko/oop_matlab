function rawImg_NonUnitvecNorm_pix5_0NN()
    iter = 100;
    avgAccVec = zeros(iter,1);
    
    for a = 1:iter
        [obj,nObj,sObj] = setExperimentParameters();
        %obj.load_Indian_Pines();
        obj.load_Indian_Pines_no_normalization(); %temporarily ONLY
        obj.selectXPixPerClassNoUnitNorm_IncludeXNeighbours(nObj);
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        sObj.unmixing(obj);
        %acc = obj.acc;

        avgAccVec(a) = obj.acc;
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
    obj.numClasses = 16;
    obj.lambda = 0.9;
end