function sumKernels()

    avgAcc = 0;
    iter = 100;
    tic;
    for a = 1:iter
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();

        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end

        cObj = CombineClass;
        cObj.kernelMatrices(obj,sunObj,svmObj);

        svmAccuracy = cObj.acc;
        avgAcc = avgAcc + svmAccuracy;
         
        fileName = fullfile('Experiments','SVM','indian_pines','results','text','svmSumKernel_5Pix_0NN_ScaledAlphas.txt');
        fileID = fopen(fileName,'a');
        fprintf(fileID,'Iteration: %d, SVM accuracy: %4.3f:\n',a,svmAccuracy);
        fclose(fileID);
    end
    
    avgAcc = avgAcc/iter;
    fileName = fullfile('Experiments','SVM','indian_pines','results','text','svmSumKernel_5Pix_0NN_ScaledAlphas.txt');
    fileID = fopen(fileName,'a');
    fprintf(fileID,'Average SVM accuracy: %4.3f:\n',avgAcc);
    fclose(fileID);
    stop = 1;
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