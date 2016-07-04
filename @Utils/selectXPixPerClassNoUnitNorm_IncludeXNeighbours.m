function selectXPixPerClassNoUnitNorm_IncludeXNeighbours(obj,nObj)
% input: indian_pines,indian_pines_gt,numBands, numPix, numNeigh, numClasses
 
uniqueClasses = unique(obj.indian_pines_gt);
[width, height] = size(obj.indian_pines_gt);
classMatrix = zeros(obj.numClasses,width,height);
% Find the different class labeles from the image pines classification map
% For each class find the (x,y) location - rows, columns of the class class label
% Store the class label (x,y) positions in classIndex varable
parfor i = 1:(obj.numClasses)
    [rowClass,colClass] = find(obj.indian_pines_gt(:,:) == uniqueClasses(i+1));
    classIndex{i} = [rowClass,colClass];
    %classMatrix(1:16,:,:) = obj.indian_pines_gt(:,:) == 1:16; -> doesnt work...
end

%Generate random training and testing indices for each class
parfor  i = 1:(obj.numClasses)
    len = size(classIndex{i},1);
    
    trainPixelsNum = obj.numPix; %trainPixelsNum = 15;
    testPixelsNum = len - trainPixelsNum;
    
    index = randperm(len,trainPixelsNum +  testPixelsNum);
    
    trainIndex = index(1:trainPixelsNum);
    testIndex = index(trainPixelsNum + 1 : trainPixelsNum +  testPixelsNum);
   
    classTrainIndex{i} = [trainIndex'];
    classTestIndex{i} = [testIndex'];
end

trainLabels = [];
testLabels = [];
parfor  i = 1:(obj.numClasses)
    trainPixels = [];
    testPixels = [];
    for  j = 1 : size(classTestIndex{i},1)
         testPix_x = classIndex{i}(classTestIndex{i}(j,1),1);
         testPix_y = classIndex{i}(classTestIndex{i}(j,1),2);
         testLabels = [testLabels; i];
         testPixels = [testPixels; testPix_x, testPix_y];
    end
    
    for  j = 1:size(classTrainIndex{i},1)
        trainPix_x = classIndex{i}(classTrainIndex{i}(j,1),1); % Random training index at position j. We 
                                                               % access the exisiting row indices from classIndex using the row value
                                                               % at the jth position from classTrainIndex.
        trainPix_y = classIndex{i}(classTrainIndex{i}(j,1),2); % Random training index at position j. We 
                                                               % access the exisiting row indices from classIndex using the row value
                                                               % at the jth position from classTrainIndex.
        trainPixels = [trainPixels; trainPix_x, trainPix_y];   % The selected row - trainPix_x and column - trainPix_y are now indices 
                                                               % of our training pixels
        trainLabels = [trainLabels; i];
    end 
    trainPixIndClass{i} = trainPixels;                         % For each class accumulate all trainPix_x and trainPix_y pixel indices
    testPixIndClassVar{i}  = testPixels;
end
obj.trainLabels = trainLabels;
%obj.testLabels = testLabels;

for i = 1:(obj.numClasses)
    trainMatrix{i} = [];
    testMatrixVar{i} = [];
    for j = 1: size(classTestIndex{i},1)
        testPixSpecVector = obj.indian_pines(testPixIndClassVar{i}(j,1),testPixIndClassVar{i}(j,2),:);
        testPixSpecVector = reshape(testPixSpecVector, 1,obj.numBands);
        testMatrixVar{i} = [testMatrixVar{i}; testPixSpecVector]; % each trainMatrix is N x d                                                      
    end

    for j = 1:size(classTrainIndex{i},1)
        trainPixSpecVector = obj.indian_pines(trainPixIndClass{i}(j,1),trainPixIndClass{i}(j,2),:);
        trainPixSpecVector = reshape(trainPixSpecVector, 1,obj.numBands);
        trainMatrix{i} = [trainMatrix{i}; trainPixSpecVector]; % each trainMatrix is N x d
    end
end
obj.trainPixIndClass = trainPixIndClass;
%obj.testPixIndClass = testPixIndClass;
obj.testPixIndClass = testPixIndClassVar;

obj.classTrainIndex = classTrainIndex;
obj.trainMatrix = trainMatrix;
obj.testMatrix = testMatrixVar;

%getNeighbours method
%[neighbours,neighboursData,testPixIndClass] = nObj.getXNeighbours(obj);

if(obj.numNeigh>0)
    obj = nObj.getXNeighbours(obj);
end

% Normalization of the training, testing and neighbourhood data
testLabels = [];
for i =1: obj.numClasses
    %obj.testMatrix{i} = [];
    testMatrixVar{i} = [];
    neighbourMatrix = [];
    for j = 1: size(obj.testPixIndClass{i},1)
            obj.testPixSpecVector = obj.indian_pines(obj.testPixIndClass{i}(j,1),obj.testPixIndClass{i}(j,2),:);
            obj.testPixSpecVector = reshape(obj.testPixSpecVector, 1,obj.numBands);
            %obj.testMatrix{i} = [obj.testMatrix{i}; obj.testPixSpecVector]; % each trainMatrix is N x d  
            testMatrixVar{i} = [ testMatrixVar{i}; obj.testPixSpecVector];
            testLabels = [testLabels; i];
    end   
    obj.testMatrix{i} = testMatrixVar{i};
    % Normalisation of the training pixels
    %obj.trainMatrix{i} = obj.normalizeX(obj.trainMatrix{i});
    %obj.trainMatrix{i} = obj.normalizeToUnitVector(obj.trainMatrix{i});
    % Normalisation of the test data:
    %obj.testMatrix{i} = obj.normalizeX(obj.testMatrix{i});
    %obj.testMatrix{i} = obj.normalizeToUnitVector(obj.testMatrix{i});

    % Neighbour data should be normalized too:
    if(obj.numNeigh > 0) 
        for j=1:size(nObj.neighboursData{i},2)  %-> 5 from 1x5 cell
            neighbourMatrix = [neighbourMatrix; nObj.neighboursData{i}{j}];
            nObj.neighboursData{i}{j} = [];
        end
        nObj.neighboursData{i} = neighbourMatrix;
        %nObj.neighboursData{i} = obj.normalizeX(nObj.neighboursData{i});
        %.neighboursData{i} = obj.normalizeToUnitVector(nObj.neighboursData{i});
    end
end
obj.testLabels = testLabels;

% The trainMatrix and the neighboursData are already normalized, so we don't
% have to normalize here anything, we just append the neighboursData
% to the trainMatrix
obj.trainData = [];
obj.testData = [];
%obj.testNeighbourData = [];
nObj.neighData = [];
for i = 1:(obj.numClasses)
    obj.trainData = [obj.trainData; obj.trainMatrix{i}(:,:)];
    obj.testData = [obj.testData; obj.testMatrix{i}(:,:)];

    if(obj.numNeigh > 0) 
        nObj.neighMatrix{i} = nObj.neighboursData{i};
        nObj.neighData = [nObj.neighData;  nObj.neighMatrix{i}];
    end
end

if(obj.numNeigh > 0) 
    nObj.testNeighLabels = zeros(size(nObj.neighData,1),1);
end
  % output: trainData,trainLabels, testData, testLabels,trainMatrix,testMatrix,NO neighbours variable,neighData,neighMatrix,testNeighLabels]
end


