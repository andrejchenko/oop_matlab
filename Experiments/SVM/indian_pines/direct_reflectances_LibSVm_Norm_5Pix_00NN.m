function direct_reflectances_LibSVm_Norm_5Pix_00NN()
    avgAcc = 0;
    iter = 100;
    avgAccVec = zeros(iter,1);

    tic;
    for a = 1:iter
        [obj,nObj,svmObj] = setExperimentParameters();
        %obj.load_Indian_Pines();
        obj.load_Indian_Pines_direct_reflectance(); %temporarily ONLY
        obj.selectXPixelPerClass_16BitVal_libSVMNormalization(nObj);
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
    save avgAccVec_svm_only_refl avgAccVec
    save avgAcc_svm_only_refl avgAcc
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