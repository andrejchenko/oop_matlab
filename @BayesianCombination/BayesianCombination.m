classdef BayesianCombination < handle
    properties
        trainData
        trainLabels
        acc
        labels
    end
    methods
        
         function method_LeeEtAll_priors_from_trainData(bObj,obj,nObj,svmObj,sunObj)
         end
         
         function method_LeeEtAll_priors_from_testData(bObj,obj,nObj,svmObj,sunObj)
             %SVM training and classification
             svmObj.svmTrainNoValidationData(obj);
             
             %% Unmixing
             sunObj.unmixingTrainData(obj);
             
             %Sum to one standardization of the alpha and alphas_validation values
             obj.sumToOneNorm(sunObj);
             obj.sumAlphaValuesPerClass(sunObj);
             
             %Generate prior class probabilities
             obj.pr_cl_prob = zeros(obj.numClasses,1);
             for i = 1:obj.numClasses
                 obj.pr_cl_prob(i) = size(obj.testMatrix{i},1)/size(obj.testData,1);
             end
         end
         
         function method_LeeEtAll_priors_from_testData_comb_testData(bObj,obj,nObj,svmObj,sunObj)
             %SVM training using train data and classification of the test data
             svmObj.svmClassification(obj);
             
             %% Unmixing
             sunObj.unmixing(obj); % unmix test data using train data as endmembers
             
             %Sum to one standardization of the alpha and alphas_validation values
             obj.sumToOneNorm(sunObj);
             obj.sumTestAlphaValuesPerClass(sunObj);
             
             %Generate prior class probabilities
             obj.pr_cl_prob = zeros(obj.numClasses,1);
             for i = 1:obj.numClasses
                 obj.pr_cl_prob(i) = size(obj.testMatrix{i},1)/size(obj.testData,1);
             end
         end
         
         function method_BenediktssonEtAll_priors_from_trainData(bObj,obj,nObj,svmObj,sunObj)
             
         end
         
         function method_BenediktssonEtAll_priors_from_testData(bObj,obj,nObj,svmObj,sunObj)
            
         end
         
    end
end
