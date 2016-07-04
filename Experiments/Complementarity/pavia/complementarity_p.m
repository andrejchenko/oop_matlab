function complementarity()
    iter = 100;
    cObj = CombineClass;
    tic;
    for a = 1:iter
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();
        obj.load_Pavia();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        
        svmClassification(svmObj,obj);
        sunObj.unmixing(obj);
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
        
        prob_values = svmObj.prob_values;
        alphas = sunObj.alphas;
        %cObj.compareClusterCentersAllPixMeanReldist(prob_values,alphas');
        e = cObj.eucledianPixelwiseDistances(prob_values,alphas');
        kd = cObj.kl_Divergence(prob_values,alphas');
        a = cObj.Bhattacharyya_angle(prob_values,alphas');
        M = [e kd a];
        corrplot(M,'varNames',{'ED','KLD','B'} );
        %cObj.klDivergence_Matlab(prob_values,alphas');
    end
    timeSpent  = toc;
end

function [obj,nObj,svmObj,sunObj] = setExperimentParameters()
    obj = Utils;
    nObj = Neighbours;
    svmObj = SVM;
    sunObj = Sunsal;
    % Parameters
    obj.numPix = 5;
    obj.numNeigh = 0;
    obj.numClasses = 9;
    obj.lambda = 0.9;
    obj.numValPixPerClass = 5;
end