function complementarity_mlr()

    iter = 100;
    cObj = CombineClass;
    tic;
    for a = 1:iter
        [obj,nObj,svmObj,sunObj] = setExperimentParameters();
        obj.load_Indian_Pines();
        obj.selectXPixPerClass_IncludeXNeighbours(nObj);
        
        load('predicted');
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
        
        prob_values = predicted;
        alphas = sunObj.alphas;
        %cObj.compareClusterCentersAllPixMeanReldist(prob_values,alphas');
        e = cObj.eucledianPixelwiseDistances(prob_values,alphas');
        kd = cObj.kl_Divergence(prob_values,alphas');
        a = cObj.Bhattacharyya_angle(prob_values,alphas');
        k = cObj.Bhattacharayya_coefficient(prob_values,alphas');
        kc = cObj.Bhattacharayya_coefficient_corr(prob_values,alphas');
        %bd = cObj.Bhattacharayya_distance_matlab(prob_values,alphas');
        M = [e kd a];
        corrplot(M,'varNames',{'ED','KLD','BA'} );
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
    obj.numClasses = 16;
    obj.lambda = 0.9;
    obj.numValPixPerClass = 5;
end