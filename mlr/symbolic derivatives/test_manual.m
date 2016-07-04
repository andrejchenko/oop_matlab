fh = @grad_funct_manual;
x_1 = [1 1 1];
beta_mat = zeros(5,3);

beta_mat(1,1) = 2;
beta_mat(1,2) = 3;
beta_mat(1,3) = 4;

y = feval(fh,x_1,beta_mat);
