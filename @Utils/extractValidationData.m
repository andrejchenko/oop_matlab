function extractValidationData(obj)

%valData = zeros(obj.numValPixPerClass,obj.numBands);
valLabels = [];
obj.valData = [];
obj.testData = [];
for  i = 1:(obj.numClasses)
    szTestData = size(obj.testData);
    len = size(obj.testMatrix{i},1);
    index = randperm(len,obj.numValPixPerClass);
    
    obj.valDataMatrix{i} = obj.testMatrix{i}(index,:);
    szTestData = size(obj.testMatrix{i});
    obj.testMatrix{i}(index,:) = [];
    %validationLabels = obj.testLabels(index);
    validationLabels = i*ones(obj.numValPixPerClass,1);
    valLabels = [valLabels; validationLabels];
    obj.testLabels(index) = [];
    szTestData = size(obj.testMatrix{i});

    obj.valData = [obj.valData; obj.valDataMatrix{i}(:,:)];
    obj.testData = [obj.testData; obj.testMatrix{i}(:,:)];
end

obj.valLabels = valLabels;
%obj.testData = obj.testMatrix;
%obj.valData =  obj.valDataMatrix;   
end