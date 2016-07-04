classdef Sunsal < handle
    properties 
       predLabels
       alphas
       alphas_validation
       cutSortedAlphas
       indexAlpha
    end
    methods
        function acc = unmixing(sObj,obj)
            %input: trainData,trainLabels,testData,testLabels, lambda
            E = obj.trainData';
            alphas = sunsal(E,obj.testData','Positivity','yes','addone','no','lambda',obj.lambda);
            %THIS IS WITH SUM TO 1 !!!!!
            %alphas = sunsal(E,obj.testData','Positivity','yes','addone','yes','lambda',obj.lambda);
            [Y,I] = max(alphas); % Hard labeling using the maximal abundance value    alphaLabels = getLabels(alphasT); % EVAL_APHA = calcAccuracy(testLabels,alphaLabels);
            %alphaLabels = [];
            label = obj.trainLabels(I);
            EVAL_APHA = obj.calcXAccuracy(obj.testLabels,label);
            acc = EVAL_APHA(1)*100;
            str = ['Sunsal Unmixing accuracy: ', num2str(acc)];
            obj.acc = acc;
            sObj.predLabels = label;
            sObj.alphas = alphas;
        end
        
        function acc = unmixing_positivity_no(sObj,obj)
            %input: trainData,trainLabels,testData,testLabels, lambda
            E = obj.trainData';
            alphas = sunsal(E,obj.testData','Positivity','no','addone','no','lambda',obj.lambda);
            %THIS IS WITH SUM TO 1 !!!!!
            %alphas = sunsal(E,obj.testData','Positivity','yes','addone','yes','lambda',obj.lambda);
            [Y,I] = max(alphas); % Hard labeling using the maximal abundance value    alphaLabels = getLabels(alphasT); % EVAL_APHA = calcAccuracy(testLabels,alphaLabels);
            %alphaLabels = [];
            label = obj.trainLabels(I);
            EVAL_APHA = obj.calcXAccuracy(obj.testLabels,label);
            acc = EVAL_APHA(1)*100;
            str = ['Sunsal Unmixing accuracy: ', num2str(acc)];
            obj.acc = acc;
            sObj.predLabels = label;
            sObj.alphas = alphas;
        end
        
        function acc = unmixingTrainData(sObj,obj)
            %input: trainData,trainLabels,testData,testLabels, lambda
            E = obj.trainData';
            %alphas = sunsal(E,obj.trainData','Positivity','yes','addone','no','lambda',obj.lambda);
            % THIS IS WITH SUM TO ONE
            alphas = sunsal(E,obj.trainData','Positivity','yes','addone','yes','lambda',obj.lambda);
            [Y,I] = max(alphas); % Hard labeling using the maximal abundance value    alphaLabels = getLabels(alphasT); % EVAL_APHA = calcAccuracy(testLabels,alphaLabels);
            sObj.alphas = alphas; % 80 x 10169 if applied to test data
        end
        
        function acc = unmixingValidationData(sObj,obj)
            %input: trainData,trainLabels,testData,testLabels, lambda
            E = obj.trainData';
            %alphas = sunsal(E,obj.trainData','Positivity','yes','addone','no','lambda',obj.lambda);
            % THIS IS WITH SUM TO ONE
            alphas = sunsal(E,obj.valData','Positivity','yes','addone','yes','lambda',obj.lambda);
            [Y,I] = max(alphas); % Hard labeling using the maximal abundance value    alphaLabels = getLabels(alphasT); % EVAL_APHA = calcAccuracy(testLabels,alphaLabels);
            sObj.alphas = alphas; % 80 x 10169 if applied to test data
        end
        
        function acc = unmixingValidationData2(sObj,obj)
            %input: trainData,trainLabels,testData,testLabels, lambda
            E = obj.trainData';
            %alphas = sunsal(E,obj.trainData','Positivity','yes','addone','no','lambda',obj.lambda);
            % THIS IS WITH SUM TO ONE
            alphas = sunsal(E,obj.valData','Positivity','yes','addone','yes','lambda',obj.lambda);
            [Y,I] = max(alphas); % Hard labeling using the maximal abundance value    alphaLabels = getLabels(alphasT); % EVAL_APHA = calcAccuracy(testLabels,alphaLabels);
            sObj.alphas_validation = alphas; % 80 x 10169 if applied to test data
        end
        
        function errorDist(sObj)
            iter = 100;
            errorMatrix = zeros(16,100);
            testLabelsMatrix = zeros(16,100);
            for a = 1:iter
                [obj,nObj] = sObj.setExperimentParams();
                obj.load_Indian_Pines();
                obj.selectXPixPerClass_IncludeXNeighbours(nObj);
                if(obj.numNeigh > 0)
                    obj.assembleXTrainData(nObj);
                end
                sObj.unmixing(obj);
                %acc = obj.acc;
                % sObj.predLabels
                % obj.testLabels %to see in which classes the unmixing
                % makes the most mistakes 
                
                for c = 1: obj.numClasses
                    testLabels = obj.testLabels(obj.testLabels == c);
                    ln = length(testLabels);
                    testLabelsMatrix(c,a) = ln;
                    predLabels = sObj.predLabels(1:ln);
                    equalVec = eq(testLabels,predLabels);
                    notEqual = equalVec(equalVec == 0);
                    len = length(notEqual);
                    errorMatrix(c,a) = len;
                    sObj.predLabels(1:ln) = []; 
                    % check new length of sObj.predLabels
                end
            end
            %sumErrorMatrix = zeros(16);
            errorPercent = errorMatrix./testLabelsMatrix;
            sumErrorPercent = sum(errorPercent,2);
            meanErrorPercent = sumErrorPercent./100;
            
            sumErrorMatrix = sum(errorMatrix,2);
            totalErrors = sum(sum(errorMatrix));
            errorContrib = sumErrorMatrix./totalErrors;
            
            save errorMatrix_sunsal1 errorMatrix
            save testLabelsMatrix_sunsal1 testLabelsMatrix
            
            save errorPercent_sunsal1 errorPercent;
            save sumErrorPercent_sunsal1 sumErrorPercent
            save meanErrorPercent_sunsal1 meanErrorPercent
            
            save sumErrorMatrix_sunsal1 sumErrorMatrix
            save totalErrors_sunsal1 totalErrors
            save errorContrib_sunsal1 errorContrib
        end
        
        function [obj,nObj] = setExperimentParams(sObj)
            obj = Utils;
            nObj = Neighbours;
            % Parameters
            obj.numPix = 5;
            obj.numNeigh = 0;
            obj.numClasses = 16;
            obj.lambda = 0.9;
        end
       
        function labels = est_y_per_class(sObj,obj,E)
            
            for i = 1:obj.numClasses
                sObj.compute_y_estimates(i,obj,E);
            end
            [min_residual,labels] = min(obj.r_perClass,[],2);
        end
        
        function compute_y_estimates(sObj,i,obj,E,r_perClass)
            alpha_idx = sObj.accessAlphas(i,obj,E);
            alphaPerClass = zeros(size(sObj.alphas,1),size(sObj.alphas,2));
            alphaPerClass(alpha_idx,:) = sObj.alphas(alpha_idx,:);
            y_est_perClass = E*alphaPerClass;
            y = obj.testData';
            %r_perClass = zeros(size(y,2),obj.numClasses); % N x numClasses
            for j = 1:size(y,2)
                r_perClass_perPixel = norm(y(:,j) - y_est_perClass(:,j));
                obj.r_perClass(j,i) = r_perClass_perPixel;
            end
            
        end
        
        function alpha_idx = accessAlphas(sObj,i,obj,E)
            alpha_idx = [];
            for j = 1:obj.numPix
                idx = j + (i-1)*obj.numPix ;
                alpha_idx = [alpha_idx; idx];
            end
        end
        
    end
end

