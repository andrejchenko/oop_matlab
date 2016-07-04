function svm_5Pix_0NN_pavia()
    avgAcc = 0;
    iter = 100;
    tic;
    for a = 1:iter
        [obj,nObj,svmObj] = setExperimentParameters();
        obj.load_Pavia();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        obj.assembleXTrainData(nObj);

        svmClassification(svmObj,obj);

        str = ['SVM accuracy: ', num2str(svmObj.accuracy(1))];
        str
        svmAccuracy = svmObj.accuracy(1);
        avgAcc = avgAcc + svmAccuracy;
        fileName = fullfile('Experiments','SVM','pavia_uni','results','text','svm_5Pix_XNN_pavia.txt');
        fileID = fopen(fileName,'a');
        fprintf(fileID,'Iteration: %d, SVM accuracy: %4.3f:\n',a,svmAccuracy);
        fclose(fileID);
    end
    timeSpent  = toc;
    avgAcc = avgAcc/iter;
    fileName = fullfile('Experiments','SVM','pavia_uni','results','text','svm_5Pix_XNN_pavia.txt');
    fileID = fopen(fileName,'a');
    fprintf(fileID,'Average SVM accuracy: %4.3f:\n',avgAcc);
    fclose(fileID);
    str = ['Average outer accuracy:', num2str(avgAcc)];
end

function [obj,nObj,svmObj] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    svmObj = SVM;
    % Parameters
    obj.numPix = 5;
    obj.numNeigh = 0;
    obj.numClasses = 9;
end