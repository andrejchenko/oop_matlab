function mlr_example()
    load fisheriris
    %Define the nominal response variable.
    sp = categorical(species);
    %Fit a nominal model to estimate the species using the flower measurements 
    %as the predictor variables.
    [B,dev,stats] = mnrfit(meas,sp);
    
    %Estimate the probability of being a certain kind of species 
    %for an iris flower having the measurements (6.2, 3.7, 5.8, 0.2).
    x = [6.2, 3.7, 5.8, 0.2];
    pihat = mnrval(B,x);
    %pihat(1) = prob_setosa; pihat(2) = prob_versicolor; 
    %pihat(3) = prob_virginica
end
