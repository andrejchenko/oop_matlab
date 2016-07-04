classdef CombineClass < handle
    properties
        trainData
        trainLabels
        acc
        labels
        
    end
    methods
        function  combineMethods(cObj,obj,nObj,svmObj,sunObj)
            
            %Sorting SVM classification probabilities
            svmClassification(svmObj,obj);
            % Inlcude the prior info even here
            load('meanErrorPercent_svm');
            onesMat = ones(length(meanErrorPercent),1);
            oldMeanErrorPercent = meanErrorPercent;
            meanErrorPercent = onesMat - meanErrorPercent;
            meanErrPerSvm = repmat(meanErrorPercent,1,length(obj.testLabels));
            clear meanErrorPercent;
            meanErrPerSvm = meanErrPerSvm';
            oldSVM_prob_values = svmObj.prob_values;
            %svmObj.prob_values = svmObj.prob_values.* meanErrPerSvm;
            svmObj.prob_values = svmObj.prob_values./ meanErrPerSvm;
            svmObj.prob_values(isnan(svmObj.prob_values(:,:))) = 0;
            [sortedSVMProb, indexSVMMat] = sort(svmObj.prob_values,2,'descend');
%             
            %Sorting alphas from Unmixing
            sunObj.unmixing(obj);
            alphas = sunObj.alphas'; % to get m x 16

            [sortedAlphas, indexAlphas] = sort(alphas,2,'descend'); % Apply the prior info on already sorted alphas for now
            load('meanErrorPercent_sunsal');
            onesMat_u = ones(length(meanErrorPercent),1);
            meanErrorPercent = onesMat_u - meanErrorPercent;
            meanErrPerSun = repmat(meanErrorPercent,1,length(obj.testLabels));
            clear meanErrorPercent;
            meanErrPerSun = meanErrPerSun';
            cutSortedAlphas = sortedAlphas(:,1:16);
            %cutSortedAlphas = cutSortedAlphas .* meanErrPerSun; % cutSortedAlphas new
            cutSortedAlphas = cutSortedAlphas ./ meanErrPerSun; % cutSortedAlphas new
            cutSortedAlphas(isnan(cutSortedAlphas(:,:))) = 0;
            
            [cutSortedAlphas, indexAlphas2] = sort(cutSortedAlphas,2,'descend'); % Sort the alphas (cutSortedAlphas) again
            
            %We only take the first 16 alpha values
            %cutSortedAlphas = sortedAlp has(:,1:16);
            %Selection of maximum from either the svm probabilities or alphas
            %Selection only from the first column since it contains the maximum prob. values and alphas
            val  = max(sortedSVMProb(:,1),cutSortedAlphas(:,1));
            
            %Track the index of the maximum - see from which vector it is coming from: the sortedSVMProb or the cutSortedAlphas
            %eqVal = isequal(val,sortedSVMProb(:,1));
            eqVec = val == sortedSVMProb(:,1);
            eqVec(eqVec == 0) = 2;
            stop = 1;
        end
        
        function  combineMethodsXTimes(cObj,obj,nObj,svmObj,sunObj)
            
            %% SVM classification
            svmClassification(svmObj,obj);
            % Prior info even here
            %-------------------------------------------------------------
%             load('meanErrorPercent_svm');
%             onesMat = ones(length(meanErrorPercent),1);
%             oldMeanErrorPercent = meanErrorPercent;
%             meanErrorPercent = onesMat - meanErrorPercent;
%             meanErrPerSvm = repmat(meanErrorPercent,1,length(obj.testLabels));
%             clear meanErrorPercent;
%             meanErrPerSvm = meanErrPerSvm';
%             oldSVM_prob_values = svmObj.prob_values;
%             %Inlcude the prior info
%             svmObj.prob_values = svmObj.prob_values.* meanErrPerSvm;
%             %svmObj.prob_values = svmObj.prob_values./ meanErrPerSvm;
%             svmObj.prob_values(isnan(svmObj.prob_values(:,:))) = 0;
%             [sortedSVMProb, indexSVMMat] = sort(svmObj.prob_values,2,'descend');
%             svmObj.sortedSVMProb = sortedSVMProb;
%             svmObj.indexSVMMat = indexSVMMat;
            %-------------------------------------------------------------
            %% Unmixing
            sunObj.unmixing(obj);
            alphas = sunObj.alphas'; % to get m x 16
            alphaMax = max(max(alphas));
            svmMax = max(max(oldSVM_prob_values));
            times = floor(svmMax/alphaMax);
            alphas = alphas * times; % re-scaling alphas

            %Prior info 
            load('meanErrorPercent_sunsal');
            onesMat_u = ones(length(meanErrorPercent),1);
            meanErrorPercent = onesMat_u - meanErrorPercent;
            meanErrPerSun = repmat(meanErrorPercent,1,length(obj.testLabels));
            clear meanErrorPercent;
            meanErrPerSun = meanErrPerSun';
            [sortedAlphas1, indexAlphas1] = sort(alphas,2,'descend'); % Apply the prior info on already sorted alphas for now
            cutSortedAlphas1 = sortedAlphas1(:,1:16); 
            
            % Apply the prior info on the (cut) sorted alphas directly and then
            % sort them again
            prioredAlphas = cutSortedAlphas1 .* meanErrPerSun;
            
            [cutSortedAlphas2, indexAlphas2] = sort(prioredAlphas,2,'descend');
            %cutSortedAlphas = cutSortedAlphas .* meanErrPerSun; % cutSortedAlphas new
            %cutSortedAlphas = cutSortedAlphas ./ meanErrPerSun; % cutSortedAlphas new
            % cutSortedAlphas(isnan(cutSortedAlphas(:,:))) = 0;
            sunObj.cutSortedAlphas = cutSortedAlphas2;
            sunObj.indexAlpha = indexAlphas2;
             %TO DO: Pointers to pointers to pointers
            
            %We only take the first 16 alpha values
            %cutSortedAlphas = sortedAlphas(:,1:16);
            %Selection of maximum from either the svm probabilities or alphas
            %Selection only from the first column since it contains the maximum prob. values and alphas
            val  = max(svmObj.sortedSVMProb(:,1), sunObj.cutSortedAlphas(:,1));
            
            %Track the index of the maximum - see from which vector it is coming from: the sortedSVMProb or the cutSortedAlphas
            %eqVal = isequal(val,sortedSVMProb(:,1));
            eqVec = val == sortedSVMProb(:,1); % 1 - svm, 0 - abundances
            % eqVec(eqVec == 0) = 2; %find()
            % sum(eqVec == 1) % 3986 pixels classified by the SVM
            % sum(eqVec == 0) % 6183 pixels classified by the unmixing
           
            stop = 1;
        end
        
        function  combineMethodsClassSelection(cObj,obj,nObj,svmObj,sunObj)
            
            %Sorting SVM classification probabilities
            svmClassification(svmObj,obj);
            % Inlcude the prior info even here
            %load('meanErrorPercent_svm');
            %onesMat = ones(length(meanErrorPercent),1);
            %oldMeanErrorPercent = meanErrorPercent;
            %meanErrorPercent = onesMat - meanErrorPercent;
            %meanErrPerSvm = repmat(meanErrorPercent,1,length(obj.testLabels));
            %clear meanErrorPercent;
            %meanErrPerSvm = meanErrPerSvm';
            %oldSVM_prob_values = svmObj.prob_values;
            %svmObj.prob_values = svmObj.prob_values.* meanErrPerSvm;
            %svmObj.prob_values = svmObj.prob_values./ meanErrPerSvm;
            svmObj.prob_values(isnan(svmObj.prob_values(:,:))) = 0;
            [sortedSVMProb, indexSVMMat] = sort(svmObj.prob_values,2,'descend');
%             
            %Sorting alphas from Unmixing
            sunObj.unmixing(obj);
            alphas = sunObj.alphas'; % to get m x 16

            [sortedAlphas, indexAlphas] = sort(alphas,2,'descend'); % Apply the prior info on already sorted alphas for now
            %load('meanErrorPercent_sunsal');
            %onesMat_u = ones(length(meanErrorPercent),1);
            %meanErrorPercent = onesMat_u - meanErrorPercent;
            %meanErrPerSun = repmat(meanErrorPercent,1,length(obj.testLabels));
            %clear meanErrorPercent;
            %meanErrPerSun = meanErrPerSun';
            cutSortedAlphas = sortedAlphas(:,1:16);
            %cutSortedAlphas = cutSortedAlphas .* meanErrPerSun; % cutSortedAlphas new
            %cutSortedAlphas = cutSortedAlphas ./ meanErrPerSun; % cutSortedAlphas new
            cutSortedAlphas(isnan(cutSortedAlphas(:,:))) = 0;
            
            %[cutSortedAlphas, indexAlphas2] = sort(cutSortedAlphas,2,'descend'); % Sort the alphas (cutSortedAlphas) again
            
            %We only take the first 16 alpha values
            %cutSortedAlphas = sortedAlphas(:,1:16);
            %Selection of maximum from either the svm probabilities or alphas
            %Selection only from the first column since it contains the maximum prob. values and alphas
            val  = max(sortedSVMProb(:,1),cutSortedAlphas(:,1));
            
            %Track the index of the maximum - see from which vector it is coming from: the sortedSVMProb or the cutSortedAlphas
            %eqVal = isequal(val,sortedSVMProb(:,1));
            eqVec = val == sortedSVMProb(:,1);
            eqVec(eqVec == 0) = 2;
            
            classSelection = [2;2;1;1;2;1;1;2;1;2;2;2;2;1;1;2];
            classesUnmix = [1;1;0;0;1;0;0;1;0;1;1;1;1;0;0;1];
            classAlphas = [1;2;5;8;10;11;12;13;16];
            
            idx_svm = find(eqVec == 1);
            classes_svm = indexSVMMat(idx_svm,1);
            %if there are values for classes which are in the class selection vector and should be instead 
            % class values from the alpha values.
            %find(classes_svm == classAlphas);

            idx_sun = find(eqVec == 2);
            classes_sun = indexAlphas(idx_sun,1);
            
            for i = 1:length(classes_svm)
                for j = 1:length(classesUnmix)
                    if(classes_svm(i) == classesUnmix(j))
                        classes_svm(i) = indexAlphas(idx_sun,1) % TO DO!!: rely on the class that is determined by the alpha values ...cutSortedAlphas
                    end
                end
            end
            
            predicted = ones(length(obj.test_labels),1)*(-1);
            predicted(idx_svm) = classes_svm;
            predicted(idx_sun) = classes_sun;
            
            stop = 1;
        end
        
        function [sortedSVMProb,indexMat] = sortMat(cObj,svmObj)
            %[] = sort(svmObj.predict_label,2);
        end
        
        function zScoreStand(cObj,sunObj)
%             meanV = mean(sunObj.alphas,2);
%             stdV = std(sunObj.alphas');
%             stdV = stdV';
%             meanV = repmat(meanV,1,size(sunObj.alphas,2));
%             stdV = repmat(stdV,1,size(sunObj.alphas,2));
%             standardizedValues = (sunObj.alphas - meanV)./stdV;
%             sunObj.alphas = standardizedValues;

            z = zscore(sunObj.alphas');
            sunObj.alphas = z';
            stop = 1;
        end
        
        function sumNormalization(cObj,sunObj,obj)
            %Sum alpha values per class and then divide each value by the
            %sum from the corresponding class
            sumAlphas = zeros(obj.numClasses,length(obj.trainLabels));
            for i = 1:obj.numClasses
                [r c] = find(obj.trainLabels == i);
                classAlphas = sunObj.alphas(r,:);
                sumed = sum(classAlphas);
                sumAlphas(i,:) = sumed;
            end
            %sunObj.alphas = sumAlphas;
            %Divide by the sum of each class
            %Find the max per class from the standatdized alpha values
            maxAlphas = zeros(obj.numClasses,length(obj.trainLabels));
            stdAlphas = [];
            for i = 1:obj.numClasses
                [r c] = find(obj.trainLabels == i);
                classAlphas = sunObj.alphas(r,:);
                sizeClass = size(classAlphas,1);
                divBySumAlphas = classAlphas./repmat(sumAlphas(i,:),sizeClass,1);
                
                var = isnan(divBySumAlphas);
                divBySumAlphas(var) = 0;
                stdAlphas = [stdAlphas;divBySumAlphas];
                %maxed = max(divBySumAlphas);
                %maxAlphas(i,:) = maxed;
            end
            %sunObj.alphas = maxAlphas;
            sunObj.alphas = stdAlphas;
        end
        
        function stacking()
        end
        
        function histEqualize(cObj,sunObj)
            imshow(sunObj.alphas);
            figure; imhist(sunObj.alphas)
            alphasEQ = histeq(sunObj.alphas);
            figure, imshow(alphasEQ)
            figure; imhist(alphasEQ)
        end
        
        function minMaxStd(cObj,sunObj)
            %maxT = max(alphas')';
            maxA = max(sunObj.alphas')';
            minA = min(sunObj.alphas')';
            minedAlphas = sunObj.alphas - repmat(minA,1,size(sunObj.alphas,2));
            stdAlphas = minedAlphas ./  repmat(maxA,1,size(sunObj.alphas,2));
            sunObj.alphas = stdAlphas;
        end
        
         function minMaxStdCorr(cObj,sunObj)
            %maxT = max(alphas')';
            maxA = max(sunObj.alphas);
            minA = min(sunObj.alphas);
            minedAlphas = sunObj.alphas - repmat(minA,size(sunObj.alphas,1),1);
            stdAlphas = minedAlphas ./  repmat(maxA,size(sunObj.alphas,1),1);
            sunObj.alphas = stdAlphas;
         end
         
         function sumToOneNormalization(cObj,sunObj)
            sumA = sum(sunObj.alphas);
            sumedToOneAlphas = sunObj.alphas ./ repmat(sumA,size(sunObj.alphas,1),1);
            sunObj.alphas = sumedToOneAlphas;
            
            sumB = sum(sunObj.alphas_validation);
            sumedToOneAlphasB = sunObj.alphas_validation ./ repmat(sumB,size(sunObj.alphas_validation,1),1);
            sunObj.alphas_validation = sumedToOneAlphasB;
         end
         function sumToOneNorm(cObj,sunObj)
            sumA = sum(sunObj.alphas);
            sumedToOneAlphas = sunObj.alphas ./ repmat(sumA,size(sunObj.alphas,1),1);
            sunObj.alphas = sumedToOneAlphas;
         end
         
         function minMax(cObj,svmObj)
            %maxT = max(alphas')';
            maxA = max(svmObj.prob_values);
            minA = min(svmObj.prob_values);
            minedProbs = svmObj.prob_values - repmat(minA,size(svmObj.prob_values,1),1);
            stdProb = minedProbs ./  repmat(maxA,size(svmObj.prob_values,1),1);
            svmObj.prob_values = stdProb;
         end
        
         function minMaxStdPerClass(cObj,sunObj,obj)
            %Alpha values per class
            alphasMinMaxPerClass = zeros(obj.numClasses,length(obj.trainLabels));
            stdAlphas = [];
            for i = 1:obj.numClasses
                [r c] = find(obj.trainLabels == i);
                classAlphas = sunObj.alphas(r,:);
                maxA = max(classAlphas);
                minA = min(classAlphas);
                sizeClass = size(classAlphas,1);
                minedAlphas = classAlphas - repmat(minA,sizeClass,1);
                %It happens that some of the maxA values are 0 and division
                %by zero produces NaN.
                stdAlphasPerClass = minedAlphas ./  repmat(maxA,sizeClass,1);
                %Workaround...
                var = isnan(stdAlphasPerClass);
                stdAlphasPerClass(var) = 0;
                stdAlphas = [stdAlphas; stdAlphasPerClass];
                %minedAlphas = sunObj.alphas(r,:) - repmat(minA,size(sunObj.alphas,1),1);
                %stdAlphas = minedAlphas ./  repmat(maxA,size(sunObj.alphas,1),1);
                % maxAlphas(i,:) = maxed;
            end
            sunObj.alphas = stdAlphas;
             
         end
         
        
        function minMaxStdSVMProb(cObj,svmObj)
            %maxT = max(alphas')';
            %maxA = max(sunObj.alphas')';
            maxPP = max(svmObj.prob_values);
            %minA = min(sunObj.alphas')';
            minPP = min(svmObj.prob_values);
            minedPP = svmObj.prob_values - repmat(minPP,size(svmObj.prob_values,1),1);
            stdProb = minedPP ./  repmat(maxPP,size(svmObj.prob_values,1),1);
            svmObj.prob_values = stdProb;
        end
        
        function baseClassification(cObj,obj,nObj,svmObj,sunObj)
            
            %% SVM basic classification
            svmTrainClassification(svmObj,obj);
            svmProbValuesBefore = svmObj.prob_values;
            
            %% Unmixing
            sunObj.unmixingTrainData(obj);
            alphas = sunObj.alphas;

            %Scaling of alphas is necesarry since they have lower
            %values then the posterior probabilities: http://stats.stackexchange.com/questions/19216/variables-are-often-adjusted-e-g-standardised-before-making-a-model-when-is
            
            %Simple X times scaling of alpha values
            %alphaMax = max(max(alphas));
            %svmMax = max(max(svmProbValuesBefore));
            %times = floor(svmMax/alphaMax);
            %alphas = alphas .* times; % re-scaling alphas  80x10169 if applied to test data
            
            %Min-max standardization of alpha values
            cObj.minMaxStd(sunObj);
            %Z-score standardization of alpha values
            %zScoreStand(cObj,sunObj);
            
            %histogram equalization
            %cObj.histEqualize(sunObj);
            %figure; imhist(sunObj.alphas)
            %figure; imhist(svmObj.prob_values)
            
            %obj.trainLabels(trainLabels)
            %sumedAlphas = zeros(obj.numClasses,length(obj.trainLabels));
            %for i = 1:obj.numClasses
            %   [r c] = find(obj.trainLabels == i);
            %   classAlphas = alphas(r,:);
            %   sumed = sum(classAlphas);
            %   sumedAlphas(i,:) = sumed;
            %end
            %sunObj.alphas = sumedAlphas';
            
            %Max alpha values per class
            maxAlphas = zeros(obj.numClasses,length(obj.trainLabels));
            for i = 1:obj.numClasses
              [r c] = find(obj.trainLabels == i);
              classAlphas = sunObj.alphas(r,:);
              maxed = max(classAlphas);
              maxAlphas(i,:) = maxed;
            end
            sunObj.alphas = maxAlphas;
        end
        
        function combineMaxClassifVal(cObj,obj,nObj,svmObj,sunObj)
            
            %% SVM classification + adding additional features as Entropy and Max posterior probabiliity
            svmTrainClassification(svmObj,obj);
            svmProbValuesBefore = svmObj.prob_values;
            
            %Add aditional features for the meta - classifier, such as maxProbability and entropy
            %SVM
            %Feature type I :
            %Multiply each posterior probability with the max posterior
            %probability
            % p(c_1|x) = p(c_1|x) * max[p(c_1|x),p(c_2|x),p(c_3|x)...p(c_16|x)]
            % p(c_2|x) =
            % p(c_2|x) * max[p(c_1|x),p(c_2|x),p(c_3|x)...p(c_16|x)] ....etc.
            %Feature type II :
            % Entropy
            % E(x) = - [p(c_1|x)*log(p(c_1|x)) + p(c_2|x)*log(p(c_2|x)) + ... + p(c_16|x)*log(p(c_16|x))] = E(pix)
            %Feature type I :
            maxProb = max(svmObj.prob_values);
            svmObj.prob_values = svmObj.prob_values .* repmat(maxProb,size(svmObj.prob_values,1),1);
            %Feature type II :
            %svmProbValuesBefore
            entropiesSVM = zeros(1,length(obj.trainLabels))';
            for i=1:size(svmObj.prob_values,1) %loop through all pixels
                entropiesSVM(i) = - sum(svmProbValuesBefore(i,:).*(log2(svmProbValuesBefore(i,:))));
            end
            %Append entropies as last feature to svmObj.prob_values
            svmObj.prob_values = [svmObj.prob_values entropiesSVM];
            
            %% Unmixing
            sunObj.unmixingTrainData(obj);
            
            %Min-max standardization of alpha values
            cObj.minMaxStdCorr(sunObj);
            
            %Max alpha values per class
            maxAlphas = zeros(obj.numClasses,length(obj.trainLabels));
            for i = 1:obj.numClasses
                [r c] = find(obj.trainLabels == i);
                classAlphas = sunObj.alphas(r,:);
                maxed = max(classAlphas);
                maxAlphas(i,:) = maxed;
            end
            sunObj.alphas = maxAlphas;
            
            %Additional Unmixing features
            %Feature type I :
            %Multiply each abundance value coming from each class with the max abundance value from all classes
            %This is not going to affect the normalized abundance values at all...
            %a(c_1|x) = a(c_1|x) * max[a(c_1|x),a(c_2|x),a(c_3|x)...a(c_16|x)]
            %a(c_2|x) = a(c_2|x) * max[a(c_1|x),a(c_2|x),a(c_3|x)...a(c_16|x)] ...etc
            %Feature type II :
            % Entropy -> remark alpha values are not probabilites, they do
            % not sum to one
            % E(x) = - [a(c_1|x)*log(a(c_1|x)) + a(c_2|x)*log(a(c_2|x)) + ... + a(c_16|x)*log(a(c_16|x))] = E(pix)
            %Feature type I :
            alphasOld = sunObj.alphas;
            maxAlphas = max(sunObj.alphas');
            alphas = sunObj.alphas'.*repmat(maxAlphas,size(sunObj.alphas',1),1);
            sunObj.alphas =  alphas';
            %Feature type II : Its not going to work for the abundance
            %values since we have many zero abundance values and the
            %entropy formula uses log2.
            %entropiesAlphas = zeros(1,length(obj.trainLabels))';
            %alphasOld = alphasOld';
            %for i=1:size(sunObj.alphas',1) %loop through all pixels
            %    entropiesAlphas(i) = - sum(alphasOld(i,:).*(log2(alphasOld(i,:))));
            %    entropiesAlphas(i) = - sum(alphasOld(i,:).*(log2(alphasOld(i,:))));
            %end
            %Append entropies as last feature to svmObj.prob_values
            %sunObj.alphas = [sunObj.alphas entropiesAlphas];
            stop = 1;
        end
        
         function combineMaxClassifValNormedAlphas(cObj,obj,nObj,svmObj,sunObj)
            
            %% SVM classification + adding additional features as Entropy and Max posterior probabiliity
            svmTrainClassification(svmObj,obj);
            svmProbValuesBefore = svmObj.prob_values;

            %% Unmixing
            sunObj.unmixingTrainData(obj);
            
            %Min-max standardization of alpha values
            %cObj.minMaxStdCorr(sunObj);
            %cObj.zScoreStand(sunObj);
            cObj.sumNormalization(sunObj,obj);
            %cObj.minMaxStdPerClass(sunObj,obj);
            
            %Max alpha values per class
            maxAlphas = zeros(obj.numClasses,length(obj.trainLabels));
            for i = 1:obj.numClasses
                [r c] = find(obj.trainLabels == i);
                classAlphas = sunObj.alphas(r,:);
                maxed = max(classAlphas);
                maxAlphas(i,:) = maxed;
            end
            sunObj.alphas = maxAlphas;
            stop = 1;
         end
        
         
         function combineMethodsWithValidationData(cObj,obj,nObj,svmObj,sunObj)
            
            %% SVM classification + adding additional features as Entropy and Max posterior probabiliity
            svmTrainValidateUsingValidationData(svmObj,obj); % here we validate the model using validation data
            % The resulting probabilities are coming from the validation data                                                                             

            %% Unmixing
            sunObj.unmixingValidationData(obj);
            
            %Min-max standardization of alpha values
            cObj.minMaxStdCorr(sunObj);
            %cObj.zScoreStand(sunObj);
            %cObj.sumNormalization(sunObj,obj);
            %cObj.minMaxStdPerClass(sunObj,obj);
            
            %Max alpha values per class
            maxAlphas = zeros(obj.numClasses,length(obj.valLabels));
            for i = 1:obj.numClasses
                [r c] = find(obj.trainLabels == i);
                classAlphas = sunObj.alphas(r,:);
                maxed = max(classAlphas);
                maxAlphas(i,:) = maxed;
            end
            sunObj.alphas = maxAlphas;
            stop = 1;
         end
         
         function combineMethodsWithValDataSum(cObj,obj,nObj,svmObj,sunObj)
             
             %% SVM classification + adding additional features as Entropy and Max posterior probabiliity
             svmTrainValidateUsingValidationData(svmObj,obj); % here we validate the model using validation data
             % The resulting probabilities are coming from the validation data
             
             %% Unmixing
             sunObj.unmixingValidationData(obj);
             
             %Sum to one standardization of alpha values
             %cObj.minMaxStdCorr(sunObj);
             %cObj.zScoreStand(sunObj);
             cObj.sumToOneNorm(sunObj);
             %cObj.minMaxStdPerClass(sunObj,obj);
             
             %Sum alpha values per class
             sumAlphas = zeros(obj.numClasses,length(obj.valLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 sumed = sum(classAlphas);
                 sumAlphas(i,:) = sumed;
             end
             sunObj.alphas = sumAlphas;
             stop = 1;
         end
         
         function combineMethodsWithValDataUsedToTrainSum(cObj,obj,nObj,svmObj,sunObj)
             
             %% SVM classification + adding additional features as Entropy and Max posterior probabiliity
             svmTrainValidateUsingValidationData(svmObj,obj); % here we validate the model using validation data
             % The resulting probabilities are coming from the validation data
             % These probabilities will be part of the meta level data -
             % svmObj.prob_values

             %% Unmix the validation data with respect to training endmember data
             sunObj.unmixingValidationData(obj);
             
             %Sum to one standardization of alpha values
             %cObj.minMaxStdCorr(sunObj);
             %cObj.zScoreStand(sunObj);
             cObj.sumToOneNorm(sunObj);
             %cObj.minMaxStdPerClass(sunObj,obj);
             
             %Sum alpha values per class
             sumAlphas = zeros(obj.numClasses,length(obj.valLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 sumed = sum(classAlphas);
                 sumAlphas(i,:) = sumed;
             end
             sunObj.alphas = sumAlphas;
             stop = 1;
             %These abundances will be the second part of the meta level
             %data
         end
         
         
         function combineMethodsWithValDataMax(cObj,obj,nObj,svmObj,sunObj)
             
             %% SVM classification + adding additional features as Entropy and Max posterior probabiliity
             svmTrainValidateUsingValidationData(svmObj,obj); % here we validate the model using validation data
             % The resulting probabilities are coming from the validation data
             
             %% Unmixing
             sunObj.unmixingValidationData(obj);
             
             %Sum to one standardization of alpha values
             %cObj.minMaxStdCorr(sunObj);
             %cObj.zScoreStand(sunObj);
             cObj.sumToOneNorm(sunObj);
             %cObj.minMaxStdPerClass(sunObj,obj);
             
             %Max alpha values per class
             maxAlphas = zeros(obj.numClasses,length(obj.valLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 maxed = max(classAlphas);
                 maxAlphas(i,:) = maxed;
             end
             sunObj.alphas = maxAlphas;
             stop = 1;
         end
         
         function combineMethodsWithValDataAvg(cObj,obj,nObj,svmObj,sunObj)
             
             %% SVM classification + adding additional features as Entropy and Max posterior probabiliity
             svmTrainValidateUsingValidationData(svmObj,obj); % here we validate the model using validation data
             % The resulting probabilities are coming from the validation data
             
             %% Unmixing
             sunObj.unmixingValidationData(obj);
             
             %Sum to one standardization of alpha values
             %cObj.minMaxStdCorr(sunObj);
             %cObj.zScoreStand(sunObj);
             cObj.sumToOneNorm(sunObj);
             %cObj.minMaxStdPerClass(sunObj,obj);
             
             %Mean alpha values per class
             avgAlphas = zeros(obj.numClasses,length(obj.valLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 meaned = mean(classAlphas);
                 avgAlphas(i,:) = meaned;
             end
             sunObj.alphas = avgAlphas;
             stop = 1;
         end
         
         function combineMethodsSVMSunal(cObj,obj,nObj,svmObj,sunObj)

             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters 
             svmObj.svmTrainUsingTrainDataTuneUsingValData(obj);
             
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             sunObj.unmixingValidationData2(obj); % unmixing the validation data for creating the validation probabiliy + abundance
             %dataset for tunning the parameters of the svm meta classifier
             
             %Sum to one standardization of the alpha and alphas_validation values
             cObj.sumToOneNormalization(sunObj);

             %Max alpha values per class
             maxAlphas = zeros(obj.numClasses,length(obj.trainLabels));
             maxAlphasVal = zeros(obj.numClasses,length(obj.valLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 maxed = max(classAlphas);
                 maxAlphas(i,:) = maxed;
                 
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas_validation(r,:);
                 maxed = max(classAlphas);
                 maxAlphasVal(i,:) = maxed;
             end
             sunObj.alphas = maxAlphas;
             sunObj.alphas_validation = maxAlphasVal;
             stop = 1;
         end
         
         function combineMethodsSVMSunalAvg(cObj,obj,nObj,svmObj,sunObj)

             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters 
             svmObj.svmTrainUsingTrainDataTuneUsingValData(obj);
             
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             sunObj.unmixingValidationData2(obj); % unmixing the validation data for creating the validation probabiliy + abundance
             %dataset for tunning the parameters of the svm meta classifier
             
             %Sum to one standardization of the alpha and alphas_validation values
             cObj.sumToOneNormalization(sunObj);

             %Avg alpha values per class
             avgAlphas = zeros(obj.numClasses,length(obj.trainLabels));
             avgAlphasVal = zeros(obj.numClasses,length(obj.valLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 averagedA = mean(classAlphas);
                 avgAlphas(i,:) = averagedA;
                 
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas_validation(r,:);
                 averagedB = mean(classAlphas);
                 avgAlphasVal(i,:) = averagedB;
             end
             sunObj.alphas = avgAlphas;
             sunObj.alphas_validation = avgAlphasVal;
             stop = 1;
         end
         
         function combineMethodsSVMSunalNoValidation(cObj,obj,nObj,svmObj,sunObj)
             
             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters
             svmObj.svmTrainNoValidationData(obj);
             
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha and alphas_validation values
             cObj.sumToOneNorm(sunObj);
             
             %Avg alpha values per class
             avgAlphas = zeros(obj.numClasses,length(obj.trainLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 averagedA = mean(classAlphas);
                 avgAlphas(i,:) = averagedA;
             end
             sunObj.alphas = avgAlphas;
             stop = 1;
         end
         
         function combineMethodsSVMUnitNormedSunalSum1NoValidation(cObj,obj,nObj,svmObj,sunObj)
             
             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters
             svmObj.svmTrainNoValidationData(obj);
             %Normalize svm probbilities - unit normalization
             %trainMatrix = normalizeX(obj,trainMatrix)
             obj.normalizeSvmProbToUnitVector(svmObj);
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha and alphas_validation values
             cObj.sumToOneNorm(sunObj);
             
             %Sum alpha values per class
             sumAlphas = zeros(obj.numClasses,length(obj.trainLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 sumed = sum(classAlphas);
                 sumAlphas(i,:) = sumed;
             end
             sunObj.alphas = sumAlphas;
             stop = 1;
         end
         
         function combineMethodsSVMUnitNormedSunalSum1AvgNoValidation(cObj,obj,nObj,svmObj,sunObj)
             
             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters
             svmObj.svmTrainNoValidationData(obj);
             %Normalize svm probbilities - unit normalization
             %trainMatrix = normalizeX(obj,trainMatrix)
             obj.normalizeSvmProbToUnitVector(svmObj);
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha and alphas_validation values
             cObj.sumToOneNorm(sunObj);
             
             %Average alpha values per class
             avgAlphas = zeros(obj.numClasses,length(obj.trainLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 meaned = mean(classAlphas);
                 avgAlphas(i,:) = meaned;
             end
             sunObj.alphas = avgAlphas;
             stop = 1;
         end
         
          function combineMethodsSVMUnitNormedSunalSum1MaxNoValidation(cObj,obj,nObj,svmObj,sunObj)
             
             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters
             svmObj.svmTrainNoValidationData(obj);
             %Normalize svm probbilities - unit normalization
             %trainMatrix = normalizeX(obj,trainMatrix)
             obj.normalizeSvmProbToUnitVector(svmObj);
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha and alphas_validation values
             cObj.sumToOneNorm(sunObj);
             
             %Max alpha values per class
             maxAlphas = zeros(obj.numClasses,length(obj.trainLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 maxed = max(classAlphas);
                 maxAlphas(i,:) = maxed;
             end
             sunObj.alphas = maxAlphas;
             stop = 1;
         end
         
         function combineMethodsSVMSunalNoValidationSum(cObj,obj,nObj,svmObj,sunObj)
             
             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters
             svmObj.svmTrainNoValidationData(obj);
             
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha and alphas_validation values
             cObj.sumToOneNorm(sunObj);
             
             %Sum alpha values per class
             sumAlphas = zeros(obj.numClasses,length(obj.trainLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 sumA = sum(classAlphas);
                 sumedAlphas(i,:) = sumA;
             end
             sunObj.alphas = sumedAlphas;
             stop = 1;
         end
         %
         function combineMethods_SVM_Distances_SunalSum1Sum_NoValidation(cObj,obj,nObj,svmObj,sunObj)
             
             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters
             svmObj.svmDistancesTrainNoValidationData(obj);
             
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha values
             cObj.sumToOneNorm(sunObj);
             
             %Sum alpha values per class
             sumAlphas = zeros(obj.numClasses,length(obj.trainLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 sumA = sum(classAlphas);
                 sumedAlphas(i,:) = sumA;
             end
             sunObj.alphas = sumedAlphas;
             stop = 1;
         end
         
         function combineMethods_SVM_Distances_SunalSum1_NoValidation(cObj,obj,nObj,svmObj,sunObj)
             
             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters
             svmObj.svmDistancesTrainNoValidationData(obj);
             %Zscore normalization of the dist_values
             svmObj.dist_values = zscore(svmObj.dist_values);
             
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha values
             cObj.sumToOneNorm(sunObj);
             stop = 1;
         end
         
         function combineMethods_SVM_Distances_SunalSum1(cObj,obj,nObj,svmObj,sunObj)
             
             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters
             svmObj.svmDistancesTrainNoValidationData(obj);

             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha values
             cObj.sumToOneNorm(sunObj);
             stop = 1;
         end
         
         function combineMethodsSVMSunalNoValidation96F(cObj,obj,nObj,svmObj,sunObj)
             
             %The resulting probabilities are coming from the training data
             %Validation probabilities are used for tuning the parameter of
             %the meta svm classifier
             %SVM meta classifier is trained with the training
             %probabilities with the tuned parameters
             svmObj.svmTrainNoValidationData(obj);
             
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha and alphas_validation values
             cObj.sumToOneNorm(sunObj);
             stop = 1;
         end
         
         function maxAlphasPerClassForTestData(cObj,obj,sunObj)
             maxAlphas = zeros(obj.numClasses,size(obj.testData,1));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 maxed = max(classAlphas);
                 maxAlphas(i,:) = maxed;
             end
             sunObj.alphas = maxAlphas;
         end
         
         function avgAlphasPerClassForTestData(cObj,obj,sunObj)
             avgAlphas = zeros(obj.numClasses,size(obj.testData,1));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 averaged = mean(classAlphas);
                 avgAlphas(i,:) = averaged;
             end
             sunObj.alphas = avgAlphas;
         end
         
         function sumAlphasPerClassForTestData(cObj,obj,sunObj)
             sumAlphas = zeros(obj.numClasses,size(obj.testData,1));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 sumed = sum(classAlphas);
                 sumAlphas(i,:) = sumed;
             end
             sunObj.alphas = sumAlphas;
         end
         
          function sumAlphasPerClassForTrainData(cObj,obj,sunObj)
             sumAlphas = zeros(obj.numClasses,size(obj.trainData,1));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 sumed = sum(classAlphas);
                 sumAlphas(i,:) = sumed;
             end
             sunObj.alphas = sumAlphas;
         end
         
        
         function svmWithValidationData(cObj,obj,nObj,svmObj,sunObj)
             
             %% SVM classification + adding additional features as Entropy and Max posterior probabiliity
             svmTrainValidateUsingValidationData(svmObj,obj); % here we validate the model using validation data
             % The resulting probabilities are coming from the validation data 
         end

        function trainSvmMetaClassifier(cObj,metaSvmObj)
            
            metaSvmObj.svmTrainMetaClassification();
        end
        
        function kernelMatrices(cObj,obj,sunObj,svmObj)
            
            numTrain = size(obj.trainData,1);
            numTest = size(obj.testData,1);
            
            %sigma = 2e-3;
            [bestG,bestC] = svmObj.selectParams(obj);
            sigma = bestG;
            rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);
            
            cmd = ['-s 0 -t 4 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            K1 =  [ (1:numTrain)' , rbfKernel(obj.trainData,obj.trainData) ];
            KK1 = [ (1:numTest)'  , rbfKernel(obj.testData,obj.trainData)  ];
            
            %% Unmixing
            sunObj.unmixingTrainData(obj);
            alphas = sunObj.alphas;
            times = 4;
            alphas = alphas .* times; %rescale alphas
            K2 = alphas;
            
            sunObjT = Sunsal;
            sunObjT.unmixing(obj);
            sunObjT.alphas = sunObjT.alphas .* times; %rescale alphas
            KK2 = sunObjT.alphas';
            
            K = K1(:,2:end) + K2;
            serial = 1 :  length(obj.trainLabels);
            extK = zeros(length(obj.trainLabels),length(obj.trainLabels) + 1);
            extK(:,1) = serial;
            extK(:,2:end) = K;
            model = svmtrain(obj.trainLabels, extK, cmd);
            
            
            KK = KK1(:,2:end) + KK2;
            serialTest = 1 :  length(obj.testLabels);
            extKK = zeros(length(obj.testLabels),length(obj.trainLabels) + 1);
            extKK(:,1) = serialTest;
            extKK(:,2:end) = KK;
            
            [predClass, acc, decVals] = svmpredict(obj.testLabels, extKK, model);
            cObj.acc = acc(1);
            
            C = confusionmat(obj.testLabels,predClass)
        end
        
        function select_max_Abun_Posterior(cObj,obj,nObj,svmObj,sunObj)
            
            svmClassification(svmObj,obj);
            %svmObj.prob_values
            svmObj.prob_values = svmObj.prob_values'; % to have 16 x 10 169 format
            
            sunObj.unmixing(obj);
            % Stretch alpha values with min-max method
            cObj.minMaxStdCorr(sunObj);
            %sunObj.alphas
            %[val I] = max(sunObj.alphas);
            
            %Select the maximum posterior or abundance value
            labels = zeros(1,length(obj.testLabels));
            for j = 1:length(obj.testLabels)
                
                [valP Ip] =  max(svmObj.prob_values(:,j));
                [valA Ia] =  max(sunObj.alphas(:,j));
                
                if(valP > valA)
                    labels(j) = Ip;
                else
                    % add  if Ia == 1:5 ->class1, if Ia == 6:10 -> class2
                    class = obj.trainLabels(Ia);
                    labels(j) = class;
                end
            end
            cObj.labels = labels';
            stop = 1;
        end
        
        function select_NormBoth_Abun_Posterior(cObj,obj,nObj,svmObj,sunObj)
            
            svmClassification(svmObj,obj);
            %svmObj.prob_values
            svmObj.prob_values = svmObj.prob_values'; % to have 16 x 10 169 format
            cObj.minMax(svmObj);
            
            
            sunObj.unmixing(obj);
            % Stretch alpha values with min-max method
            cObj.minMaxStdCorr(sunObj);
            %sunObj.alphas
            %[val I] = max(sunObj.alphas);
            
            %Select the maximum posterior or abundance value
            labels = zeros(1,length(obj.testLabels));
            for j = 1:length(obj.testLabels)
                
                [valP Ip] =  max(svmObj.prob_values(:,j));
                [valA Ia] =  max(sunObj.alphas(:,j));
                
                if(valP > valA)
                    labels(j) = Ip;
                else
                    % add  if Ia == 1:5 ->class1, if Ia == 6:10 -> class2
                    class = obj.trainLabels(Ia);
                    labels(j) = class;
                end
            end
            cObj.labels = labels';
            stop = 1;
        end
        
        function select_Avg_Abun_Posterior(cObj,obj,nObj,svmObj,sunObj)
            
            svmClassification(svmObj,obj);
            %svmObj.prob_values
            svmObj.prob_values = svmObj.prob_values'; % to have 16 x 10 169 format

            sunObj.unmixing(obj);
            % Stretch alpha values with min-max method
            cObj.minMaxStdCorr(sunObj);
            %sunObj.alphas
            %[val I] = max(sunObj.alphas);
            
            %AVG alpha values per class
            avgAlphas = zeros(obj.numClasses,length(obj.testLabels)); % here we use the test pixels directly
            for i = 1:obj.numClasses
                [r c] = find(obj.trainLabels == i);
                classAlphas = sunObj.alphas(r,:);
                averaged = mean(classAlphas);
                avgAlphas(i,:) = averaged;
            end
            sunObj.alphas = avgAlphas;
            
            labels = zeros(1,length(obj.testLabels));
            for j = 1:length(obj.testLabels)
                
                [valP Ip] =  max(svmObj.prob_values(:,j));
                [valA Ia] =  max(sunObj.alphas(:,j));
                
                if(valP > valA)
                    labels(j) = Ip;
                else
                    % add  if Ia == 1:5 ->class1, if Ia == 6:10 -> class2
                    class = obj.trainLabels(Ia);
                    labels(j) = class;
                end
            end
            cObj.labels = labels';
            stop = 1;
        end
        
        
        function class = belongs(cObj,Ia)
            
        end
        
        function consensusRule(cObj,obj,nObj,svmObj,sunObj,w)
            svmClassification(svmObj,obj);
            %svmObj.prob_values
            svmObj.prob_values = svmObj.prob_values'; % to have 16 x 10 169 format
            
            sunObj.unmixing(obj);
            % Stretch alpha values with min-max method
            cObj.minMaxStdCorr(sunObj);
            
            %Max alpha values per class
            maxAlphas = zeros(obj.numClasses,length(obj.testLabels)); % here we use the test pixels directly
            for i = 1:obj.numClasses
                [r c] = find(obj.trainLabels == i);
                classAlphas = sunObj.alphas(r,:);
                maxed = max(classAlphas);
                maxAlphas(i,:) = maxed;
            end
            sunObj.alphas = maxAlphas;
            
            %Here we need all class posteriors for each test pixel &
            %The class abundance values for each test pixel 
            sumed = w * svmObj.prob_values + (1-w) * sunObj.alphas;
            [val labels] = max(sumed);
            cObj.labels = labels';
            stop = 1;
        end
        
        function consensusRuleAvgAbun(cObj,obj,nObj,svmObj,sunObj,w)
            svmClassification(svmObj,obj);
            %svmObj.prob_values
            svmObj.prob_values = svmObj.prob_values'; % to have 16 x 10 169 format
            
            sunObj.unmixing(obj);
            % Stretch alpha values with min-max method
            cObj.minMaxStd(sunObj);
            
            %AVG alpha values per class
            avgAlphas = zeros(obj.numClasses,length(obj.testLabels)); % here we use the test pixels directly
            for i = 1:obj.numClasses
                [r c] = find(obj.trainLabels == i);
                classAlphas = sunObj.alphas(r,:);
                averaged = mean(classAlphas);
                avgAlphas(i,:) = averaged;
            end
            sunObj.alphas = avgAlphas;
            
            %Here we need all class posteriors for each test pixel &
            %The class abundance values for each test pixel
            sumed = w * svmObj.prob_values + (1-w) * sunObj.alphas;
            [val labels] = max(sumed);
            cObj.labels = labels';
            stop = 1;
        end
        
        function generateMetaLevelData(cObj,obj)
            L1_trainLabels = [];
            L1_testLabels = [];
            
            L2_trainLabels = [];
            L2_testLabels = [];
            
            L1_trainData = [];
            L1_testData  = [];
            
            L2_trainData = [];
            L2_testData  = [];
            
            for i = 1:obj.numClasses
                L1_trainMatrix{i} = obj.trainMatrix{i}(1:5,:);
                L1_trainData = [L1_trainData; L1_trainMatrix{i}];
                label = i*ones(5,1);
                L1_trainLabels = [L1_trainLabels; label];
                
                L1_testMatrix{i} = obj.trainMatrix{i}(6:10,:);
                L1_testData = [L1_testData; L1_testMatrix{i}];
                L1_testLabels = [L1_testLabels; label];

                L2_trainMatrix{i} = obj.trainMatrix{i}(6:10,:);
                L2_trainData = [L2_trainData;L2_trainMatrix{i}];
                L2_trainLabels = [L2_trainLabels; label];
                
                L2_testMatrix{i} = obj.trainMatrix{i}(1:5,:);
                L2_testData = [L2_testData; L2_testMatrix{i}];
                L2_testLabels = [L2_testLabels; label];
            end
            
            obj.L1_trainMatrix = L1_trainMatrix;
            obj.L1_trainLabels = L1_trainLabels;
            obj.L1_trainData = L1_trainData;
            
            obj.L1_testMatrix =  L1_testMatrix;
            obj.L1_testLabels =  L1_testLabels;
            obj.L1_testData   =  L1_testData;
            
            obj.L2_trainMatrix = L2_trainMatrix;
            obj.L2_trainLabels = L2_trainLabels;
            obj.L2_trainData = L2_trainData;
            
            obj.L2_testMatrix =  L2_testMatrix;
            obj.L2_testLabels =  L2_testLabels;
            obj.L2_testData = L2_testData;
        end
    end
end

