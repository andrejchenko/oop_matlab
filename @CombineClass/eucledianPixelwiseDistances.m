function euclDistances = eucledianPixelwiseDistances(cObj,prob,alphas)
distances = [];
    for i = 1: size(prob,1)
        v1 = prob(i,:);
        v2 = alphas(i,:);
        %eD = sqrt(sum((v1 - v2) .^ 2));
        %diff = (v1 - v2) .^ 2;
        %std_diff = diff./v1;
        eD = sqrt(sum((v1 - v2) .^ 2));
        distances = [distances; eD];
    end
    euclDistances = distances;
    save euclDistances euclDistances
    mean_eucl_dist = mean(euclDistances);
    save mean_eucl_dist mean_eucl_dist
    %D = pdist2(X,Y,'minkowski',P)
end