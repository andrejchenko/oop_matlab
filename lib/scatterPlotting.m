function scatterPlotting()

    [obj,nObj,svmObj,sunObj] = setExperimentParameters();
     obj.load_Indian_Pines();

    allSet = obj.indian_pines(:,:,1:50);
    YAll = reshape(allSet,145*145,50);
    plotmatrix(YAll)
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