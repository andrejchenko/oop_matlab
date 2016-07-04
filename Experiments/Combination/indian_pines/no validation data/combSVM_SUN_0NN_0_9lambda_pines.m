function combSVM_SUN_0NN_0_9lambda_pines

    [obj,nObj,svmObj,sunObj] = setExperimentParameters();

    obj.load_Indian_Pines();
    obj.selectXPixPerClass_IncludeXNeighbours(nObj);
    if(obj.numNeigh > 0)
        obj.assembleXTrainData(nObj);
    end
    
    cObj = CombineClass;
    % cObj.combineMethods(obj,nObj,svmObj,sunObj);
    % cObj.combineMethodsClassSelection(obj,nObj,svmObj,sunObj);
    cObj.combineMethodsXTimes(obj,nObj,svmObj,sunObj);
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
end