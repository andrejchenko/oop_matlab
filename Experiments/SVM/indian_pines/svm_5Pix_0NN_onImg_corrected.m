function svm_5Pix_0NN_onImg_corrected()
    avgAcc = 0;
    iter = 100;
    avgAccVec = zeros(iter,1);

    tic;
    for a = 1:iter
        [obj,nObj,svmObj] = setExperimentParameters();
        obj.load_Indian_Pines_corrected();
        %obj.load_Indian_Pines_no_normalization(); %temporarily ONLY
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        svmClassification(svmObj,obj);

        str = ['SVM accuracy: ', num2str(svmObj.accuracy(1))];
        str
        svmAccuracy = svmObj.accuracy(1);
        avgAccVec(a) = svmAccuracy;
    end
    timeSpent  = toc;
    avgAcc = mean(avgAccVec);
    save avgAccVec_svm_only avgAccVec
    save avgAcc_svm_only avgAcc
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