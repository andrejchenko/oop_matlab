function svmKernelClassify()

    avgAcc = 0;
    iter = 100;
    tic;
    for a = 1:iter
        [obj,nObj,svmObj] = setExperimentParameters();
         obj.load_Indian_Pines();
         obj.selectXPixPerClass_IncludeXNeighbours(nObj);
         svmObj.svmKernelClassify(obj);
         
         svmAccuracy = svmObj.accuracy(1);
         avgAcc = avgAcc + svmAccuracy;
         
        fileName = fullfile('Experiments','SVM','indian_pines','results','text','svmKernel_5Pix_0NN.txt');
        fileID = fopen(fileName,'a');
        fprintf(fileID,'Iteration: %d, SVM accuracy: %4.3f:\n',a,svmAccuracy);
        fclose(fileID);
    end
    
    avgAcc = avgAcc/iter;
    fileName = fullfile('Experiments','SVM','indian_pines','results','text','svmKernel_5Pix_0NN.txt');
    fileID = fopen(fileName,'a');
    fprintf(fileID,'Average SVM accuracy: %4.3f:\n',avgAcc);
    fclose(fileID);
    stop = 1; 
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