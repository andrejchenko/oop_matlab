function bayesClassifCombination_Lee_5Pix_0NN()
avgAccVec = zeros(50,21);
    iter = 50;
    c = 1;
    for w = 0:0.05:1 %exponential weight
        for a = 1:iter
            [obj,nObj,svmObj,sunObj] = setExperimentParameters();
            obj.load_Indian_Pines();
            obj.selectXPixPerClass_IncludeXNeighbours(nObj);
            
            %obj.extractValidationData();
            if(obj.numNeigh > 0)
                obj.assembleXTrainData(nObj);
            end
            bObj = BayesianCombination;
            bObj.method_LeeEtAll_priors_from_testData_comb_testData(obj,nObj,svmObj,sunObj);
            sunObj.alphas = sunObj.alphas'; % to have it both as 80x16

            denom = (sunObj.alphas.^w).*(svmObj.prob_values.^(1-w));
            for col = 1:size(sunObj.alphas,2) %column loop
                denom(:,col) = denom(:,col)./obj.pr_cl_prob(col);
            end
            
            [maxValue, idx] = max(denom');
            predicted = idx';
            eval = obj.calcXAccuracy(obj.testLabels,predicted);
            acc = eval(1)*100;
            avgAccVec(a,c) = acc;
        end
        c = c+1;
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