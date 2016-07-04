%https://en.wikipedia.org/wiki/Bhattacharyya_angle
%https://en.wikipedia.org/wiki/Bhattacharyya_distance
function angles = Bhattacharyya_angle(cObj,prob,alphas)
angles = [];
    for i = 1: size(prob,1)
        
        v1 = prob(i,:); 
        v2 = alphas(i,:);

        bc = 0;
        for j = 1:size(v1,2)
            a = sqrt(v1(j)*v2(j));
            bc = bc + a;
        end
        angle = acos(bc); % or use acosd
        angle = radtodeg(angle);
        angles = [angles; angle];
    end
    save angles angles
    mean_angle = mean(angles);
    save mean_angle mean_angle
end
