classdef Utils < handle
    properties 
    %Indian Pines properties
    indian_pines_gt  
    indian_pines
    numBands
    
    %paviaU_gt
    %pavia_gtStruct
    
    %Experiment related properties
    numPix
    numNeigh
    numClasses
    lambda
    
    %Training and testing data properties
    trainData
    trainLabels
    testData 
    testLabels
    trainMatrix
    testMatrix
  
    testPixSpecVector
    
    %Auxiliary indices
    trainPixIndClass
    testPixIndClass
    classTrainIndex
    
    acc
    
    numValPixPerClass
    valDataMatrix
    valData
    valLabels
    
    pr_cl_prob
    
    L1_trainMatrix
    L1_trainLabels
    
    L1_testMatrix
    L1_testLabels
    
    L2_trainMatrix
    L2_trainLabels
    
    L2_testMatrix
    L2_testLabels
    
    L1_trainData;
    L1_testData;
    
    L2_trainData;
    L2_testData;
    
    %Residuals from each test pixel per class
    r_perClass
    
    end
    methods
        function load_Indian_Pines(obj)
            indian_pines_raw = multibandread('\data\datasets\AVIRIS_Indian_Pines\raw\aviris', [145 145 220], 'uint16', 0, 'bsq', 'ieee-le');
            %samples = 145
            %lines   = 145
            %bands   = 220

            Xcoor = 1:145;
            Ycoor = 1:145;
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are working with the radiance values, not reflecance values...so
            % instead of: indian_pines_scaled = indian_pines_raw/10000; % to get the reflectance values
            indian_pines_scaled = indian_pines_raw/10000; % to get the reflectance values
            % I can try: RV = (SDV-1000) / 500.
            % indian_pines_scaled = (indian_pines_raw - 1000)/500;
            % bands = setdiff(1:220,[1:4 103:113 148:166]); from Rob
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are not using: 1,33,97 and 161 band.
            % I have to remove the 31st and 97th band additionally
            bands = setdiff(1:200,[1:4 33:33 97:97 103:113 148:166]); % remove certain noisy bands
            numBands = size(bands,2);
            indian_pines = indian_pines_scaled(:,:,bands);

            %indian_pines_gtStruct = load('E:\Projects\Matlab\data\datasets\AVIRIS_Indian_Pines\Indian_pines_gt.mat');
            indian_pines_gtStruct = load('Indian_pines_gt.mat');
            indian_pines_gt = indian_pines_gtStruct.indian_pines_gt;
            
            obj.indian_pines_gt = indian_pines_gt;
            obj.indian_pines = indian_pines;
            obj.numBands = numBands;
        end
        
        function load_Indian_Pines_corrected(obj)
            indian_pines_raw = load('D:\Projects\Matlab\data\datasets\AVIRIS_Indian_Pines\Indian_pines_corrected.mat');
            %samples = 145
            %lines   = 145
            %bands   = 220

            Xcoor = 1:145;
            Ycoor = 1:145;
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are working with the radiance values, not reflecance values...so
            % instead of: indian_pines_scaled = indian_pines_raw/10000; % to get the reflectance values
            
            
            %Here we use directly the reflectance values, we dont have to
            %divide by 10000
            indian_pines_scaled = indian_pines_raw.indian_pines_corrected; % to get the reflectance values
            % I can try: RV = (SDV-1000) / 500.
            % indian_pines_scaled = (indian_pines_raw - 1000)/500;
            % bands = setdiff(1:220,[1:4 103:113 148:166]); from Rob
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are not using: 1,33,97 and 161 band.
            % I have to remove the 31st and 97th band additionally
            I = setdiff(1:220,[104:108 150:163 220:220]);
            list = [1 2 3 4 33 97 103 104 105 106 107 108 109 110 111 112 113 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 201 202 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220];
            for i = 1 : length(list)
                [r,c] = find(I == list(i))
                I(c) = [];
            end
            bands = I;
            %bandsOnCorr = setdiff(1:200,[1:4 33:33 97:97 103:108 148:144,145:147]); % remove certain noisy bands
            %bands = setdiff(1:200,[1:4 33:33 97:97 103:113 148:166]); % remove certain noisy bands
            numBands = size(bands,2);
            indian_pines = indian_pines_scaled(:,:,bands);

            %indian_pines_gtStruct = load('E:\Projects\Matlab\data\datasets\AVIRIS_Indian_Pines\Indian_pines_gt.mat');
            indian_pines_gtStruct = load('Indian_pines_gt.mat');
            indian_pines_gt = indian_pines_gtStruct.indian_pines_gt;
            
            obj.indian_pines_gt = indian_pines_gt;
            obj.indian_pines = indian_pines;
            obj.numBands = numBands;
        end
        
        function load_Indian_Pines_no_normalization(obj)
            indian_pines_raw = multibandread('\data\datasets\AVIRIS_Indian_Pines\raw\aviris', [145 145 220], 'uint16', 0, 'bsq', 'ieee-le');
            
            %samples = 145
            %lines   = 145
            %bands   = 220

            Xcoor = 1:145;
            Ycoor = 1:145;
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are working with the radiance values, not reflecance values...so
            indian_pines_scaled = indian_pines_raw; 
            % I can try: RV = (SDV-1000) / 500.
            % indian_pines_scaled = (indian_pines_raw - 1000)/500;
            % bands = setdiff(1:220,[1:4 103:113 148:166]); from Rob
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are not using: 1,33,97 and 161 band.
            % I have to remove the 31st and 97th band additionally
            bands = setdiff(1:200,[1:4 33:33 97:97 103:113 148:166]); % remove certain noisy bands
            numBands = size(bands,2);
            indian_pines = indian_pines_scaled(:,:,bands);

            %indian_pines_gtStruct = load('E:\Projects\Matlab\data\datasets\AVIRIS_Indian_Pines\Indian_pines_gt.mat');
            indian_pines_gtStruct = load('Indian_pines_gt.mat');
            indian_pines_gt = indian_pines_gtStruct.indian_pines_gt;
            
            obj.indian_pines_gt = indian_pines_gt;
            obj.indian_pines = indian_pines;
            obj.numBands = numBands;
        end
        
        function load_Indian_Pines_direct_reflectance(obj)
            %indian_pines_raw = multibandread('\data\datasets\AVIRIS_Indian_Pines\raw\aviris', [145 145 220], 'uint16', 0, 'bsq', 'ieee-le');
            indian_pines_refl = load('D:\Projects\Matlab\data\datasets\AVIRIS_Indian_Pines\Indian_220_Vera.mat'); 
            indian_pines_refl =  indian_pines_refl.Inidian220;
            %samples = 145
            %lines   = 145
            %bands   = 220

            Xcoor = 1:145;
            Ycoor = 1:145;
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are working with the radiance values, not reflecance values...so
            indian_pines_scaled = indian_pines_refl; 
            % I can try: RV = (SDV-1000) / 500.
            % indian_pines_scaled = (indian_pines_raw - 1000)/500;
            % bands = setdiff(1:220,[1:4 103:113 148:166]); from Rob
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are not using: 1,33,97 and 161 band.
            % I have to remove the 31st and 97th band additionally
            bands = setdiff(1:200,[1:4 33:33 97:97 103:113 148:166]); % remove certain noisy bands
            numBands = size(bands,2);
            indian_pines = indian_pines_scaled(:,:,bands);

            %indian_pines_gtStruct = load('E:\Projects\Matlab\data\datasets\AVIRIS_Indian_Pines\Indian_pines_gt.mat');
            indian_pines_gtStruct = load('Indian_pines_gt.mat');
            indian_pines_gt = indian_pines_gtStruct.indian_pines_gt;
            
            obj.indian_pines_gt = indian_pines_gt;
            obj.indian_pines = indian_pines;
            obj.numBands = numBands;
        end
        
        function load_Pavia(obj)
            %pavia = multibandread('\data\datasets\AVIRIS_Pavia\PaviaU', [610 340 103], 'uint16', 0, 'bsq', 'ieee-le');
            pavia = load('D:\Projects\Matlab\data\datasets\AVIRIS_Pavia\PaviaU.mat');
            %samples = 610
            %lines   = 340
            %bands   = 103
            % Pavia univeristy has in total 115 bans, this image contains only the
            % bands without the noise already
            Xcoor = 1:610;
            Ycoor = 1:340;
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are working with the radiance values, not reflecance values...so
            % instead of: indian_pines_scaled = indian_pines_raw/10000; % to get the reflectance values
            pavias_scaled = pavia.paviaU/10000; % to get the reflectance values
            % I can try: RV = (SDV-1000) / 500.
            % indian_pines_scaled = (indian_pines_raw - 1000)/500;
            % bands = setdiff(1:220,[1:4 103:113 148:166]); from Rob
            % On the calibrated data from Purdue University on this page: https://purr.purdue.edu/publications/1947/supportingdocs
            % they are not using: 1,33,97 and 161 band.
            % I have to remove the 31st and 97th band additionally
            %bands = setdiff(1:200,[1:4 33:33 97:97 103:113 148:166]); % remove certain noisy bands
            %numBands = size(bands,2);
            %indian_pines = pavias_scaled(:,:,bands);
            numBands = 103;
            %indian_pines_gtStruct = load('E:\Projects\Matlab\data\datasets\AVIRIS_Indian_Pines\Indian_pines_gt.mat');
            pavia_gtStruct = load('D:\Projects\Matlab\data\datasets\AVIRIS_Pavia\PaviaU_gt.mat');
            paviaU_gt = pavia_gtStruct.paviaU_gt;
            
            %obj.paviaU_gt = paviaU_gt;
            %obj.pavia_gtStruct = pavia_gtStruct;
            %obj.numBands = numBands;
            
            obj.indian_pines_gt = paviaU_gt;
            obj.indian_pines = pavias_scaled;
            obj.numBands = numBands;
            
            %The selectXPixPerClass_IncludeXNeighbours method uses the indian_pines_gt and indian_pines properties of the object
            %So in order not to create additional function doing the same I
            % things I will fill in the indian_pines_gt and indian_pines properties with pavia data
            % output: paviaU_gt,pavias_scaled,numBands
        end
        
        
        function trainMatrix = normalizeX(obj,trainMatrix)
            tempTrainMatrix = [];
            for j = 1: size(trainMatrix,1)
                pix = trainMatrix(j,:);
                sq_sum = 0;
                for z = 1: size(pix,2)
                    sq_sum = sq_sum + pix(z)*pix(z);
                end
                %sq_sum = sq_sum/size(pix,2);
                pixNorm = sqrt(sq_sum);
                pixUnit = pix./pixNorm;
                tempTrainMatrix = [tempTrainMatrix;pixUnit];
            end
            trainMatrix = tempTrainMatrix;
        end
        
        function trainMatrix = normalizeToUnitVector(obj,trainMatrix)
            tempTrainMatrix = [];
            for j = 1: size(trainMatrix,1) %trainMatrix is 5x164, 5 row vectors
                pix = trainMatrix(j,:);
                pixUnit = pix/norm(pix);
                tempTrainMatrix = [tempTrainMatrix;pixUnit];
            end
            trainMatrix = tempTrainMatrix;
        end
        
        function trainMatrix = libSVM_normalization(obj,trainMatrix)
            data = trainMatrix;
            %(data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2))
            tempData = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
            trainMatrix = tempData;
        end
        
        function trainMatrix = normalizeSvmProbToUnitVector(obj,svmObj)
            tempProbValues = [];
            for j = 1: size(svmObj.prob_values,1) %trainMatrix is 5x164, 5 row vectors
                pix = svmObj.prob_values(j,:);
                pixUnit = pix/norm(pix);
                tempProbValues = [tempProbValues;pixUnit];
            end
            svmObj.prob_values = tempProbValues;
        end
        
        function assembleXTrainData(obj1,nObj)
            % input trainData, trainMatrix, neighMatrix, neighData, numClasses
            trainData = [];
            trainLabels = [];
            
            for i = 1:(obj1.numClasses)
                trainData = [trainData; obj1.trainMatrix{i}(:,:)];
                for j = 1:size(obj1.trainMatrix{i},1)
                    trainLabels = [trainLabels; i];
                end
                trainData = [trainData; nObj.neighMatrix{i}];
                
                for j = 1:size(nObj.neighMatrix{i},1)
                    trainLabels = [trainLabels; i];
                end
                trainMatrix{i} = [obj1.trainMatrix{i};nObj.neighMatrix{i}];
            end
            obj1.trainMatrix = trainMatrix;
            obj1.trainData = trainData;
            obj1.trainLabels = trainLabels;
            %output: trainData,trainLabels
        end
        
        function EVAL = calcXAccuracy(obj,ACTUAL,PREDICTED)
            % Barnan Das
            % http://www.mathworks.com/matlabcentral/fileexchange/37758-performance-measures-for-classification/content/Evaluate.m
            % This fucntion evaluates the performance of a classification model by
            % calculating the common performance measures: Accuracy, Sensitivity,
            % Specificity, Precision, Recall, F-Measure, G-mean.
            % Input: ACTUAL = Column matrix with actual class labels of the training
            %                 examples
            %        PREDICTED = Column matrix with predicted class labels by the
            %                    classification model
            % Output: EVAL = Row matrix with all the performance measures
            
            
            idx = (ACTUAL()==1);
            
            p = length(ACTUAL(idx));
            n = length(ACTUAL(~idx));
            N = p+n;
            
            tp = sum(ACTUAL(idx)==PREDICTED(idx));
            tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
            fp = n-tn;
            fn = p-tp;
            
            tp_rate = tp/p;
            tn_rate = tn/n;
            
            accuracy = (tp+tn)/N;
            sensitivity = tp_rate;
            specificity = tn_rate;
            precision = tp/(tp+fp);
            recall = sensitivity;
            f_measure = 2*((precision*recall)/(precision + recall));
            gmean = sqrt(tp_rate*tn_rate);
            
            EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
        end
        
        function sumToOneNorm(obj,sunObj)
            sumA = sum(sunObj.alphas);
            sumedToOneAlphas = sunObj.alphas ./ repmat(sumA,size(sunObj.alphas,1),1);
            sunObj.alphas = sumedToOneAlphas;
        end
         
        function sumAlphaValuesPerClass(obj,sunObj)
            %Sum alpha values per class
             sumAlphas = zeros(obj.numClasses,length(obj.trainLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 sumed = sum(classAlphas);
                 sumAlphas(i,:) = sumed;
             end
             sunObj.alphas = sumAlphas;
        end
        
        function sumTestAlphaValuesPerClass(obj,sunObj)
            %Sum alpha values per class
             sumAlphas = zeros(obj.numClasses,length(obj.testLabels));
             for i = 1:obj.numClasses
                 [r c] = find(obj.trainLabels == i);
                 classAlphas = sunObj.alphas(r,:);
                 sumed = sum(classAlphas);
                 sumAlphas(i,:) = sumed;
             end
             sunObj.alphas = sumAlphas;
        end
       
        function r = multiplyBy(obj,n)
            r = [obj.Value] * n;
            obj.Value = r;
        end
    end
end