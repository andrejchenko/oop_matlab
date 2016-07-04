function mlr_Sub_Classify_5Pix_0NN()
    iter = 100;
    avgAccVec = zeros(iter,1);
    o_acc_vec = zeros(iter,1);
    for i = 1:iter
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();

        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        
        mObj = MLR;
        mObj.trainMLR_Sub(obj);
        [acc,o_acc] = mObj.predictMLR_Sub(obj);
        avgAccVec(i) = acc; 
        o_acc_vec(i) = o_acc;
    end

    avgAcc = mean(avgAccVec);
    save avgAccVec_mlr_sub avgAccVec
    save avgAcc_mlr_sub avgAcc
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