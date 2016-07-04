function compareClusterCentersAllPixMeanReldist(cObj,prob,alphas)
    mean_P = mean(prob);
    mean_A = mean(alphas);
    dist = abs(mean_P - mean_A);
    relD = dist./mean_P;
    allPixMean_relDistances = relD;
    save allPixMean_relDistances
end

