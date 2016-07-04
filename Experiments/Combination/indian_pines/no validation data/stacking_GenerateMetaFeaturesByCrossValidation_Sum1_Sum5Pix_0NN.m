function stacking_GenerateMetaFeaturesByCrossValidation_Sum1_Sum5Pix_0NN
 avgAccVec = zeros(50,1);
    iter = 50;
    for a = 1:iter
        
        %% TRAINING PHASE
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();

        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        obj.extractValidationData();
        if(obj.numNeigh > 0)
            obj.assembleXTrainData(nObj);
        end
        cObj = CombineClass;
        %Generate meta level data
        cObj.generateMetaLevelData(obj);
        %Train first level models: SVM_0 and E_0
        objL1 = Utils;
        objL1.trainData =   obj.L1_trainData;
        objL1.trainLabels = obj.L1_trainLabels;
        objL1.testData =    obj.L1_testData;
        objL1.testLabels =  obj.L1_testLabels;
        svmObj.svmClassification(objL1);
        L1_prob_values = svmObj.prob_values;
        
        objL2 = Utils;
        objL2.trainData = obj.L2_trainData;
        objL2.trainLabels = obj.L2_trainLabels;
        objL2.testData =    obj.L2_testData;
        objL2.testLabels =  obj.L2_testLabels;
        svmObj.svmClassification(objL2);
        L2_prob_values = svmObj.prob_values;
        
        objL1.lambda = 0.9;
        objL1.numClasses = 16;
        sunObj.unmixing(objL1);
        L1_alphas = sunObj.alphas;
        cObj.sumToOneNorm(sunObj); %sum to 1 normalization of the abundances
        objL1.sumTestAlphaValuesPerClass(sunObj)%sum abundances per class
        L1_alphas = sunObj.alphas;
        
        objL2.lambda = 0.9;
        objL2.numClasses = 16;
        sunObj.unmixing(objL2);
        L2_alphas = sunObj.alphas;
        cObj.sumToOneNorm(sunObj); %sum to 1 normalization of the abundances
        objL2.sumTestAlphaValuesPerClass(sunObj)%sum abundances per class
        L2_alphas = sunObj.alphas;
        
        alphas = [L1_alphas L2_alphas];
        sunObj.alphas = alphas;

        alphas = alphas';
        prob_values = [L1_prob_values;L2_prob_values];
        
        cObj.trainData = [alphas prob_values]; %160 x 32
        cObj.trainLabels = [objL1.testLabels;objL2.testLabels]; %160
        %Train the meta SVM model here 
        svmObj.trainSvmMetaClassifier(cObj,obj); %svmMetaModel  modelMeta   % obj as input is not necesarry here

        %Train the SVM model on the whole: training data now without
        %splitting up to L1 and L2 subsets
        svmObj.svmTrainClassification(obj); %svmModel model
        
        %% TEST Phase
        metaSvmObj = SVM;
        acc = metaSvmObj.svmMetaTestOnTestExtendedSetSum(obj,sunObj,cObj);
        avgAccVec(a) = acc; 
    end
    save avgAccVec avgAccVec
    avgAcc = mean(avgAccVec);
    save avgAcc avgAcc
end

function [obj,nObj,svmObj,sunObj] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    svmObj = SVM;
    sunObj = Sunsal;
    % Parameters
    obj.numPix = 10;
    obj.numNeigh = 0;
    obj.numClasses = 16;
    obj.lambda = 0.9;
    obj.numValPixPerClass = 5;
end