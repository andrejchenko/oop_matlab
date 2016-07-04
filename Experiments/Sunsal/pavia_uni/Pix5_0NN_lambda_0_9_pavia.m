function Pix5_0NN_lambda_0_9_pavia()
tic;
    avgUnmixing = 0;
    iter = 100;
    for a = 1:100
        [obj,nObj,sObj] = setExperimentParameters();
        obj.load_Pavia();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        obj.assembleXTrainData(nObj);
        sObj.unmixing(obj);
        %acc = obj.acc;

        avgUnmixing = avgUnmixing + obj.acc;
        fileName = fullfile('Experiments','Sunsal','pavia_uni','results','text','Unmixing_5Pix_lambda_0_dot_9_5Neighbours_pavia.txt');
        fileID = fopen(fileName,'a');
        fprintf(fileID,'Iteration: %d, Sunsal unmixing accuracy: %4.3f with fixed lambda: %4.3f\n',a,obj.acc,obj.lambda);
        fclose(fileID);
    end

    avgUnmixing = avgUnmixing/iter;
    str = ['Average sunsal unmixing accuracy: ', num2str(avgUnmixing)];
    str
        fileName = fullfile('Experiments','Sunsal','pavia_uni','results','text','Unmixing_5Pix_lambda_0_dot_9_5Neighbours_pavia.txt');
        fileID = fopen(fileName,'a');
        fprintf(fileID,'Average Sunsal unmixing accuracy: %4.3f',avgUnmixing);
        fclose(fileID);
timeSpent = toc;
end


function [obj,nObj,sObj] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    sObj = Sunsal;
    % Parameters
    obj.numPix = 5;
    obj.numNeigh = 5;
    obj.numClasses = 9;
    obj.lambda = 0.9;
end