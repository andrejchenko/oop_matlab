function consensusRule_5Pix_0NN_0_9lambda

    avgUnmixing = 0;
    iter = 100;
    for w = 0:0.1:1
        for a = 1:iter
            [obj,nObj,svmObj,sunObj] = setExperimentParameters();

            obj.load_Indian_Pines();
            obj.selectXPixPerClass_IncludeXNeighbours(nObj);
            if(obj.numNeigh > 0)
                obj.assembleXTrainData(nObj);
            end

            cObj = CombineClass;
            cObj.consensusRule(obj,nObj,svmObj,sunObj,w);

            EVAL_APHA = obj.calcXAccuracy(obj.testLabels,cObj.labels);
            acc = EVAL_APHA(1)*100;

            avgUnmixing = avgUnmixing + acc;
            fileName = fullfile('Experiments','Combination','indian_pines','results','text','consensusRule_5Pix_0NN_0_9lambda_w.txt');
            fileID = fopen(fileName,'a');
            fprintf(fileID,'Iteration: %d, Combined accuracy: %4.3f with fixed lambda: %4.3f\n',a,acc);
            fclose(fileID);
            stop1 = 1;
        end

        avgUnmixing = avgUnmixing/iter;
        fileName = fullfile('Experiments','Combination','indian_pines','results','text','consensusRule_5Pix_0NN_0_9lambda_w.txt');
        fileID = fopen(fileName,'a');
        fprintf(fileID,'Average combined accuracy: %4.3f with w: %4.3f\n',avgUnmixing,w);
        fclose(fileID);
        stop2 = 1;
    end

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