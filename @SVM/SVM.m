classdef SVM < handle
    properties 
       predict_label
       accuracy
       prob_values
       dist_values
       prob_values_validation
       bestG
       bestC
       sortedSVMProb
       indexSVMMat
    end
    methods
        function svmClassification(svmObj,obj)
            %input: trainData,trainLabels, testData, testLabels
            tic;
            % Select the optimal parameters: gamma -g and cots -c from the
            % lowest cross validation error value
            [bestG,bestC] = svmObj.selectParams(obj);
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            model = libsvmtrain(obj.trainLabels, obj.trainData, cmd);
            save svm_model_01_12_2015 model
            % Use the SVM model to classify the data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, obj.testData, model, '-b 1'); % run the SVM model on the test data
            time = toc;
            %save svmClassification_10_11_2015_5Pix_train_Rest_test_10_cv trainData  trainLabels testData testLabels time predict_label accuracy prob_values
            save svmModel  model
            %If the true labels of obj.testLabels are used in the
            %libsvmpredict, then only the accuracy output makes sense
            %If dummy test labels are used as  obj.testLabels in the
            %libsvmpredict, then the accuracy output doesnt make sens and
            %the rest: predict_label and prob_values make sense
            svmObj.predict_label = predict_label; 
            svmObj.accuracy = accuracy;
            svmObj.prob_values = prob_values;
            svmObj.bestG = bestG;
            svmObj.bestC = bestC;
            %output: predict_label, accuracy, prob_values
        end
        
        function acc = svmClassificationAcc(svmObj,obj)
            %input: trainData,trainLabels, testData, testLabels
            tic;
            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = svmObj.selectParams(obj);
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            model = libsvmtrain(obj.trainLabels, obj.trainData, cmd);
            save svm_model_01_12_2015 model
            % Use the SVM model to classify the data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, obj.testData, model, '-b 1'); % run the SVM model on the test data
            time = toc;
            %save svmClassification_10_11_2015_5Pix_train_Rest_test_10_cv trainData  trainLabels testData testLabels time predict_label accuracy prob_values
            save svmModel  model
            %If the true labels of obj.testLabels are used in the
            %libsvmpredict, then only the accuracy output makes sense
            %If dummy test labels are used as  obj.testLabels in the
            %libsvmpredict, then the accuracy output doesnt make sens and
            %the rest: predict_label and prob_values make sense
            svmObj.predict_label = predict_label; 
            svmObj.accuracy = accuracy;
            svmObj.prob_values = prob_values;
            svmObj.bestG = bestG;
            svmObj.bestC = bestC;
            acc = accuracy(1);
            %output: predict_label, accuracy, prob_values
        end
        
        function svmTrainClassification(svmObj,obj)
            %input: trainData,trainLabels, testData, testLabels
            tic;
            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = svmObj.selectParams(obj);
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            model = libsvmtrain(obj.trainLabels, obj.trainData, cmd);
            %save svm_model_08_03_2016 model
            % Use the SVM model to classify the same training data .....
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.trainLabels, obj.trainData, model, '-b 1'); % run the SVM model on the test data
            time = toc;
            %save svmClassification_10_11_2015_5Pix_train_Rest_test_10_cv trainData  trainLabels testData testLabels time predict_label accuracy prob_values
            save svmModel  model
            svmObj.predict_label = predict_label;
            svmObj.accuracy = accuracy;
            svmObj.prob_values = prob_values;
            svmObj.bestG = bestG;
            svmObj.bestC = bestC;
            %output: predict_label, accuracy, prob_values
        end
        
        function svmTrainValidateUsingValidationData(svmObj,obj)
            %input: trainData,trainLabels, testData, testLabels

            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = svmObj.selectParams(obj); % using the initial training data without using the validation data
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            model = libsvmtrain(obj.trainLabels, obj.trainData, cmd); %train the model using training data without using the validation data here
            %save svm_model_08_03_2016 model
            % Use the SVM model to classify the VALIDATION data now.....
            valDummy = zeros(size(obj.valData,1),1);
            %CHECK THE OUTPUT accuracy now
            [predict_label, accuracy, prob_values] = libsvmpredict(valDummy, obj.valData, model, '-b 1'); % run the SVM model on the validation data
            %[predict_label, accuracy, prob_values] = libsvmpredict(obj.valLabels, obj.valData, model, '-b 1'); % run the SVM model on the validation data
            
            save svmModel  model
            %svmObj.predict_label = predict_label;
           % svmObj.accuracy = accuracy;
            svmObj.prob_values = prob_values;
            %svmObj.bestG = bestG;
            %svmObj.bestC = bestC;
            %output: predict_label, accuracy, prob_values
        end
        
        function svmTrainUsingTrainAndValidationData(svmObj,obj)
            %Merge train and validation data
            obj.trainData = [obj.trainData; obj.valData];
            obj.trainLabels = [obj.trainLabels; obj.valLabels];
            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = svmObj.selectParams(obj); % using the train and validation data for training and for parameters tuning
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            modelVal = libsvmtrain(obj.trainLabels, obj.trainData, cmd); %train the model using training data without using the validation data here
            save svmModelVal  modelVal
        end
        
         function svmTrainUsingTrainDataTuneUsingValData(svmObj,obj)
            %input: trainData,trainLabels, testData, testLabels

            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = svmObj.selectParams(obj); % using the initial training data without using the validation data
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            model = libsvmtrain(obj.trainLabels, obj.trainData, cmd); %train the model using training data without using the validation data here
            %save svm_model_08_03_2016 model
            % Use the SVM model to classify the VALIDATION data now.....
            trainDummy = zeros(size(obj.trainData,1),1);
            %CHECK THE OUTPUT accuracy now
            [predict_label, accuracy, prob_values] = libsvmpredict(trainDummy, obj.trainData, model, '-b 1'); % run the SVM model on the train data
            %[predict_label, accuracy, prob_values] = libsvmpredict(obj.trainLabels, obj.trainData, model, '-b 1'); % run the SVM model on the train data
            
            valDummy = zeros(size(obj.valData,1),1);
            [predict_labelV, accuracyV, prob_valuesV] = libsvmpredict(valDummy, obj.valData, model, '-b 1'); % run the SVM model on the validation data
            
            save svmModel  model
            %svmObj.predict_label = predict_label;
           % svmObj.accuracy = accuracy;
            svmObj.prob_values = prob_values;
            svmObj.prob_values_validation = prob_valuesV;
            %svmObj.bestG = bestG;
            %svmObj.bestC = bestC;
            %output: predict_label, accuracy, prob_values
         end
        
         function svmTrainNoValidationData(svmObj,obj)
             %input: trainData,trainLabels, testData, testLabels
             
             % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
             [bestG,bestC] = svmObj.selectParams(obj); % using the initial training data without using the validation data
             
             % Train the LIB_SVM with the optimum parameters
             % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
             
             %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
             cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
             %cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q '];
             model = libsvmtrain(obj.trainLabels, obj.trainData, cmd); %train the model using training data without using the validation data here
             %save svm_model_08_03_2016 model
             
             trainDummy = zeros(size(obj.trainData,1),1);
             %CHECK THE OUTPUT accuracy now
             [predict_label, accuracy, prob_values] = libsvmpredict(trainDummy, obj.trainData, model, '-b 1'); % run the SVM model on the train data
             %[predict_label, accuracy, prob_values] = libsvmpredict(obj.trainLabels, obj.trainData, model, '-b 1'); % run the SVM model on the train data
             %Obtain the distances to hyperplanes instead of output probabilities
             %[predict_label, accuracy, dist_values] = libsvmpredict(obj.testLabels, obj.testData, model); % run the SVM model on the test data
             save svmModel  model
             %svmObj.predict_label = predict_label;
             % svmObj.accuracy = accuracy;
             svmObj.prob_values = prob_values;
             %svmObj.bestG = bestG;
             %svmObj.bestC = bestC;
             %output: predict_label, accuracy, prob_values
         end
         
          function svmDistancesTrainNoValidationData(svmObj,obj)
             %input: trainData,trainLabels, testData, testLabels
             
             % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
             [bestG,bestC] = svmObj.selectParams(obj); % using the initial training data without using the validation data
             
             % Train the LIB_SVM with the optimum parameters
             % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
             
             %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
             %cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
             cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q '];
             %cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q '];
             model = libsvmtrain(obj.trainLabels, obj.trainData, cmd);
             %save svm_model_08_03_2016 model
             
             trainDummy = zeros(size(obj.trainData,1),1);
             %CHECK THE OUTPUT accuracy now
             %[predict_label, accuracy, prob_values] = libsvmpredict(trainDummy, obj.trainData, model, '-b 1'); % run the SVM model on the train data
             [predict_label, accuracy, dist_values] = libsvmpredict(trainDummy, obj.trainData, model);
             %[predict_label, accuracy, prob_values] = libsvmpredict(obj.trainLabels, obj.trainData, model, '-b 1'); % run the SVM model on the train data
             %Obtain the distances to hyperplanes instead of output probabilities
             %[predict_label, accuracy, dist_values] = libsvmpredict(obj.testLabels, obj.testData, model); % run the SVM model on the test data
             save svmModel  model
             %svmObj.predict_label = predict_label;
             % svmObj.accuracy = accuracy;
             svmObj.dist_values = dist_values;
             %output: predict_label, accuracy, prob_values
         end
        
        function trainSvmMetaClassifier(metaSvmObj,cObj,obj) % obj as input is not necesarry here
            %input: trainData,trainLabels, testData, testLabels
            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = metaSvmObj.selectParams(cObj);
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            modelMeta = libsvmtrain(cObj.trainLabels, cObj.trainData, cmd);
            metaSvmObj.bestG = bestG;
            metaSvmObj.bestC = bestC;
            save svmMetaModel  modelMeta
            
            %output: predict_label, accuracy, prob_values
        end
        
        function trainSvmMetaValDataForTuning(metaSvmObj,cObj,obj,svmObj)
            %input: trainData,trainLabels, testData, testLabels
            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = metaSvmObj.selectParams(cObj);
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            modelMeta = libsvmtrain(obj.trainLabels, svmObj.prob_values, cmd);
            save svmMetaModel  modelMeta
            
            %output: predict_label, accuracy, prob_values
        end
        
         function trainSvmMetaValDataForTuningExtendedSet(metaSvmObj,cObj,obj,svmObj,sunObj)
            %input: trainData,trainLabels, testData, testLabels
            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = metaSvmObj.selectParams(cObj);
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            %Extended set: abundances from the trained data + probabilities from the trained data
            extendedSet = [sunObj.alphas' svmObj.prob_values];
            modelMeta = libsvmtrain(obj.trainLabels, extendedSet, cmd);
            save svmMetaModel  modelMeta
            
            %output: predict_label, accuracy, prob_values
         end
         
         function trainSvmMetaWithExtendedSetFromValData(metaSvmObj,cObj,obj,svmObj,sunObj)
            %input: trainData,trainLabels, testData, testLabels
            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = metaSvmObj.selectParams(cObj);
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            %Extended set: abundances from the validation data + probabilities from the validation data
            extendedSet = [sunObj.alphas' svmObj.prob_values];
            modelMeta = libsvmtrain(cObj.trainLabels, extendedSet, cmd);
            save svmMetaModel  modelMeta
            %output: predict_label, accuracy, prob_values
         end
        
         function trainSvmMetaUsingExtendedSet(metaSvmObj,cObj,obj,svmObj,sunObj)
            %input: trainData,trainLabels, testData, testLabels
            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = metaSvmObj.selectParams(cObj);
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            %Extended set: abundances from the trained data + probabilities from the trained data
            %extendedSet = [sunObj.alphas' svmObj.prob_values];
            modelMeta = libsvmtrain(cObj.trainLabels, cObj.trainData, cmd);
            save svmMetaModel  modelMeta
            
            %output: predict_label, accuracy, prob_values
        end

        function svmMetaTest(metaSvmObj,obj)
           
            load svmMetaModel
            % Use the SVM model to classify the data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, obj.testData, modelMeta, '-b 1'); % run the SVM model on the test data
            metaSvmObj.predict_label = predict_label;
            metaSvmObj.accuracy = accuracy;
            metaSvmObj.prob_values = prob_values;
        end
        
        function svmMetaTestOnProb(metaSvmObj,obj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to probability test data using the frist SVM model: model
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            %2. Use the SVM meta classifier model: modelMeta to classify the probability test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, test_prob_values, modelMeta, '-b 1');
            stop = 1;
            
            metaSvmObj.testOutputFrom2SVMS(obj);
        end
        
        function testOutputFrom2SVMS(svmObj, obj)
            %Test for validation data
            load svmMetaModel
            load svmModel
            valDummy = zeros(size(obj.valData,1),1);
            %1. Convert spectral validation feature data to probability validation data using the frist SVM model: model
            [val_predict_label, val_accuracy, val_prob_values] = libsvmpredict(valDummy, obj.valData, model, '-b 1');
            [val2_predict_label, val2_accuracy, val2_prob_values] = libsvmpredict(obj.valLabels, obj.valData, model, '-b 1');
            %2. Use the SVM meta classifier model: modelMeta to classify the probability validation data
            [val3_predict_label, val3_accuracy, val3_prob_values] = libsvmpredict(obj.valLabels, val_prob_values, modelMeta, '-b 1');
            
            %Test for train data
            trainDummy = zeros(size(obj.trainData,1),1);
            %1. Convert spectral train feature data to probability validation data using the frist SVM model: model
            [train_predict_label, train_accuracy, train_prob_values] = libsvmpredict(trainDummy, obj.trainData, model, '-b 1');
            [train2_predict_label, train2_accuracy, train2_prob_values] = libsvmpredict(obj.trainLabels, obj.trainData, model, '-b 1');
            %2. Use the SVM meta classifier model: modelMeta to classify the probability validation data
            [train3_predict_label, train3_accuracy, train3_prob_values] = libsvmpredict(obj.trainLabels, train_prob_values, modelMeta, '-b 1');
            stop = 1;
        end
        function acc = svmMetaTestOnProbWithExtendedSet(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to probability test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            cObj.maxAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
        end
        
        function acc = svmMetaTestOnProbWithExtendedSetAvg(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to probability test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            cObj.avgAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
        end
        
        function acc = svmMetaTestOnTestExtendedSetSum(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to probability test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            sunObj.unmixing(obj); %unmixing of test data with respect to whole training data
            %Sum to 1 Normalization:
            cObj.sumToOneNorm(sunObj);
            cObj.sumAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
        end
        
        function acc = svmMetaTestOnTestExtendedSetUsingNewModelSum(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            %load svmModel
            load svmModelVal
            %1. Convert spectral test feature data to probability test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, modelVal, '-b 1');
            
            %The obj.trainData already contains: train data + validation
            %data
            sunObj.unmixing(obj); %unmixing of test data with respect to (training data+validationdata)
            %Sum to 1 Normalization:
            cObj.sumToOneNorm(sunObj);
            cObj.sumAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
        end
        
        function acc = svmMetaTestOnFeatureTestSetUsingNewModelSum(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to probability test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            
            %The obj.trainData already contains: train data + validation
            %data
            sunObj.unmixing(obj); %unmixing of test data with respect to (training data+validationdata)
            %Sum to 1 Normalization:
            cObj.sumToOneNorm(sunObj);
            cObj.sumAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            %test2FeatureSet = [sunObj.alphas' test_prob_values];
            test2FeatureSet = [max(sunObj.alphas)' max(test_prob_values')'];
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, test2FeatureSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
        end
        
         function acc = svmMetaTestOnTestExtendedSetMax(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to probability test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            %Sum to 1 Normalization:
            cObj.sumToOneNorm(sunObj);
            cObj.maxAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
         end
        
         function acc = svmMetaTestOnTestExtendedSetAvg(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to probability test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            %Sum to 1 Normalization:
            cObj.sumToOneNorm(sunObj);
            cObj.avgAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
        end
        
         function acc = svmMetaTestOnProbWithExtendedSetSum(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to probability test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            cObj.sumAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            %Normalize the test extended set
            %testExtendedSet = zscore(testExtendedSet);
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
         end
         
          function acc = svmMetaTestOnDistancesWithExtendedSetSum(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to distnaces test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_dist_values] = libsvmpredict(testDummy, obj.testData, model);
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            %Sum to one normalization of alpha values
            cObj.sumToOneNorm(sunObj);
            %Sum alpha values per class
            cObj.sumAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_dist_values]; %(16 abundances + 120 distances) = 136 test feature vectors
            %Normalize the test extended set
            %testExtendedSet = zscore(testExtendedSet);
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
          end
         
         function acc = svmMetaTestOnNormedDistancesWithExtendedSetSum(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to distnaces test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_dist_values] = libsvmpredict(testDummy, obj.testData, model);
            test_dist_values = zscore(test_dist_values);
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            %Sum to one normalization of alpha values
            cObj.sumToOneNorm(sunObj);
            %Sum alpha values per class
            %cObj.sumAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_dist_values]; %(80 sum to 1 normalized abundances + 120  normalized distances) = 200 test feature vectors
            %Normalize the test extended set
            %testExtendedSet = zscore(testExtendedSet);
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
         end

          function acc = svmMetaTestOnDistancesWithExtendedSet(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to distnaces test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_dist_values] = libsvmpredict(testDummy, obj.testData, model);
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            %Sum to one normalization of alpha values
            cObj.sumToOneNorm(sunObj);
            %Sum alpha values per class
            %cObj.sumAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_dist_values]; %(80 sum to 1 normalized abundances + 120 distances) = 200 test feature vectors
            %Normalize the test extended set
            %testExtendedSet = zscore(testExtendedSet);
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
         end
          
         function acc = svmMetaTest_ProbUnitNormed_AbunSum1Sum(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to:
            % Rrobability test data normalized: Unit Normalization
            % Abundance test data normalized: Sum to 1 and Sum per class
            % afterwards
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            metaSvmObj.prob_values = test_prob_values;
            obj.normalizeSvmProbToUnitVector(metaSvmObj);
            test_prob_values = metaSvmObj.prob_values;
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            %Normalize abundances: Sum to 1
            cObj.sumToOneNorm(sunObj);
            cObj.sumAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
         end
         
         function acc = svmMetaTest_ProbUnitNormed_AbunSum1Avg(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to:
            % Rrobability test data normalized: Unit Normalization
            % Abundance test data normalized: Sum to 1 and Sum per class
            % afterwards
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            metaSvmObj.prob_values = test_prob_values;
            obj.normalizeSvmProbToUnitVector(metaSvmObj);
            test_prob_values = metaSvmObj.prob_values;
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            %Normalize abundances: Sum to 1
            cObj.sumToOneNorm(sunObj);
            cObj.avgAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
         end
         
         function acc = svmMetaTest_ProbUnitNormed_AbunSum1Max(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to:
            % Rrobability test data normalized: Unit Normalization
            % Abundance test data normalized: Sum to 1 and Sum per class
            % afterwards
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            metaSvmObj.prob_values = test_prob_values;
            obj.normalizeSvmProbToUnitVector(metaSvmObj);
            test_prob_values = metaSvmObj.prob_values;
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            %Normalize abundances: Sum to 1
            cObj.sumToOneNorm(sunObj);
            cObj.maxAlphasPerClassForTestData(obj,sunObj); % now we have 16 x 80
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
         end
        
          function acc = svmMetaTestOnProbWithExtendedSet96F(metaSvmObj,obj,sunObj,cObj)
            load svmMetaModel
            load svmModel
            %1. Convert spectral test feature data to probability test data + abundance test data
            testDummy = zeros(size(obj.testData,1),1);
            [test_predict_label, test_accuracy, test_prob_values] = libsvmpredict(testDummy, obj.testData, model, '-b 1');
            sunObj.unmixing(obj); %unmixing of test data with respect to training data
            testExtendedSet = [sunObj.alphas' test_prob_values];
            
            %2. Use the SVM meta classifier model to classify the probability + abundance test data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, testExtendedSet, modelMeta, '-b 1');
            acc = accuracy(1);
            stop = 1;
        end
        
        function svmMetaTestWithValidationData(metaSvmObj,obj)
           
            load svmMetaModel
            % Use the SVM model to classify the data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.valLabels, obj.valData, modelMeta, '-b 1'); % run the SVM model on the test data
            metaSvmObj.predict_label = predict_label;
            metaSvmObj.accuracy = accuracy;
            metaSvmObj.prob_values = prob_values;
        end
        
        function [bestG,bestC] = selectParams(svmObj,obj)
            % input: trainData,trainLabels
            bestcv = 0;
            log2c = -1:1:20;
            log2g = -5:1:5;
            
            cGridLength = length(log2c);
            gGridLength = length(log2g);
            
            for indexc = 1:cGridLength,
                clc;
                %fprintf('Iteration %i of %i...',indexc,cGridLength);
                for indexg = 1:gGridLength,
                    cmd = ['-q -v 5 -c ', num2str(2^log2c(indexc),2), ' -g ', num2str(2^log2g(indexg)), '  -b 1'];
                    cv = libsvmtrain(obj.trainLabels, obj.trainData, cmd);
                    if (cv >= bestcv),
                        bestcv = cv; bestC = 2^log2c(indexc); bestG = 2^log2g(indexg);
                    end
                end
            end
            %output: [bestG,bestC]
        end
        
        function [bestG,bestC] = selectParamsVal(svmObj,obj)
            % input: trainData,trainLabels
            bestcv = 0;
            log2c = -1:1:20;
            log2g = -5:1:5;
            
            cGridLength = length(log2c);
            gGridLength = length(log2g);
            
            for indexc = 1:cGridLength,
                clc;
                %fprintf('Iteration %i of %i...',indexc,cGridLength);
                for indexg = 1:gGridLength,
                    cmd = ['-q -v 10 -c ', num2str(2^log2c(indexc),2), ' -g ', num2str(2^log2g(indexg)), '  -b 1'];
                    cv = libsvmtrain(obj.valLabels, obj.valData, cmd);
                    if (cv >= bestcv),
                        bestcv = cv; bestC = 2^log2c(indexc); bestG = 2^log2g(indexg);
                    end
                end
            end
            %output: [bestG,bestC]
        end
        
        function errorDist(svmObj)
            iter = 100;
            errorMatrix = zeros(16,100);
            testLabelsMatrix = zeros(16,100);
            for a = 1:iter
                [obj,nObj] = svmObj.setExperimentParams();
                obj.load_Indian_Pines();
                obj.selectXPixPerClass_IncludeXNeighbours(nObj);
                if(obj.numNeigh > 0)
                    obj.assembleXTrainData(nObj);
                end
                svmClassification(svmObj,obj);
                %acc = obj.acc;
                % sObj.predLabels
                % obj.testLabels %to see in which classes the unmixing
                % makes the most mistakes 
                
                for c = 1: obj.numClasses
                    testLabels = obj.testLabels(obj.testLabels == c);
                    ln = length(testLabels);
                    testLabelsMatrix(c,a) = ln;
                    predLabels = svmObj.predict_label(1:ln);
                    %predLabels = sObj.predLabels(1:ln);
                    equalVec = eq(testLabels,predLabels);
                    notEqual = equalVec(equalVec == 0);
                    len = length(notEqual);
                    errorMatrix(c,a) = len;
                    svmObj.predict_label(1:ln) = []; 
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
            
            save errorMatrix_svm1 errorMatrix
            save testLabelsMatrix_svm1 testLabelsMatrix
            
            save errorPercent_svm1 errorPercent;
            save sumErrorPercent_svm1 sumErrorPercent
            save meanErrorPercent_svm1 meanErrorPercent
            
            save sumErrorMatrix_svm1 sumErrorMatrix
            save totalErrors_svm1 totalErrors
            save errorContrib_svm1 errorContrib
        end
        
        function svmKernelClassify(svmObj,obj)
            
            %obj.trainLabels, obj.trainData
            %obj.testLabels, obj.testData
            numTrain = size(obj.trainData,1);
            numTest = size(obj.testData,1);

            %sigma = 2e-3;
            [bestG,bestC] = svmObj.selectParams(obj);
            sigma = bestG;
            rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);
                    
            cmd = ['-s 0 -t 4 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            %cmd = ['-s 0 -t 4 ', '-q ', ' -b 1'];
            
            %# compute kernel matrices between every pairs of (train,train) and
            %# (test,train) instances and include sample serial number as first column
            K =  [ (1:numTrain)' , rbfKernel(obj.trainData,obj.trainData) ];
            KK = [ (1:numTest)'  , rbfKernel(obj.testData,obj.trainData)  ];
            
            %# train and test
            %model = svmtrain(obj.trainLabels, K, '-t 4');
            model = svmtrain(obj.trainLabels, K, cmd);
            [predClass, acc, decVals] = svmpredict(obj.testLabels, KK, model);
            svmObj.accuracy = acc;
            %# confusion matrix
            C = confusionmat(obj.testLabels,predClass)
        end

        function svmTrainOnly(svmObj,obj)
            % Select the optimum parameters: gamma -g and cots -c from the higest cross validation accuracy
            [bestG,bestC] = svmObj.selectParams(obj);
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ', ' -b 1'];
            model = libsvmtrain(obj.trainLabels, obj.trainData, cmd);
            save svm_model_only model
        end
        
        function svmClassifyOnly(svmObj,obj)
            load svm_model_only
         % Use the SVM model to classify the data
            [predict_label, accuracy, prob_values] = libsvmpredict(obj.testLabels, obj.testData, model, '-b 1'); % run the SVM model on the test data

            svmObj.predict_label = predict_label; 
            svmObj.accuracy = accuracy;
            svmObj.prob_values = prob_values;
        end
        
        
        
        function [obj,nObj] = setExperimentParams(svmObj)
            obj = Utils;
            nObj = Neighbours;
            % Parameters
            obj.numPix = 5;
            obj.numNeigh = 0;
            obj.numClasses = 16;
            obj.lambda = 0.9;
        end
    end
end