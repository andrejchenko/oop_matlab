%http://stackoverflow.com/questions/19972878/matlab-code-to-compare-two-histograms
function k = Bhattacharayya_coefficient(cObj,X1,X2)
    k = sum(sqrt(X1(:)).*sqrt(X2(:)));
end