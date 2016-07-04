function bayesClassifCombination_5Pix_0NN_usingOnlyTrainData()
avgAccVec = [];
    iter = 50;
    for a = 1:iter
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();
        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        
        %obj.extractValidationData();
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        bObj = BayesianCombination;
        bObj.method_BenediktssonEtAll_priors_from_testData(obj,nObj,svmObj,sunObj);
        
        sunObj.alphas = sunObj.alphas'; % to have it both as 80x16
        
        %Generate prior class probabilities 
        obj.pr_cl_prob = zeros(obj.numClasses,1);
        for i = 1:obj.numClasses
            obj.pr_cl_prob(i) = size(obj.testMatrix{i},1)/size(obj.testData,1);
        end
        
        for w = 0:0.05:1 %exponential weight
            denom = (sunObj.alphas.^w).*(svmObj.prob_values.^(1-w));
            for col = 1:size(sunObj.alphas,2) %column loop
                denom(:,col) = denom(:,col)./obj.pr_cl_prob(col);
            end
            
            [maxValue, idx] = max(denom');
            predicted = idx';
            eval = obj.calcXAccuracy(obj.trainLabels,predicted);
            acc = eval(1)*100;
        end
        
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
    obj.numValPixPerClass = 5;
end