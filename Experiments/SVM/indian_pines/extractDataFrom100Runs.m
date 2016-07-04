function extractDataFrom100Runs()
    iter = 100;

    tic;
    for a = 1:iter
        [obj,nObj,svmObj] = setExperimentParameters();
        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        
    end
end

function [obj,nObj,svmObj] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    svmObj = SVM;
    % Parameters
    obj.numPix = 5;
    obj.numNeigh = 0;
    obj.numClasses = 16;
end