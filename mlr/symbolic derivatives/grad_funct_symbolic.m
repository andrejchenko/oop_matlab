function grad_funct_symbolic()
    %beta = sym('b', [1 3]);
    x_1 = sym('x', [1 3]);

    %beta_mat = sym('b_l', [5 3]);
    beta_mat = sym('b', [5 3]);

    %expr = log(exp(beta*x_1')/(1+sum(exp(beta_mat * x_1'))));
    expr = log(exp(beta_mat(1,:)*x_1')/(1+sum(exp(beta_mat * x_1'))))

    %grad = gradient(expr, beta);
    %grad_1 = gradient(expr, beta(1));


    grad = gradient(expr, beta_mat(1,1)); % w.r.t. b_1_1
    ht = matlabFunction(grad);
    
    x_1 = [1 1 1];
    beta_mat = zeros(5,3);

    beta_mat(1,1) = 2;
    beta_mat(1,2) = 3;
    beta_mat(1,3) = 4;

    y = feval(ht,beta_mat(1,1),beta_mat(1,2),beta_mat(1,3),...
                 beta_mat(2,1),beta_mat(2,2),beta_mat(2,3),...
                 beta_mat(3,1),beta_mat(3,2),beta_mat(3,3),...
                 beta_mat(4,1),beta_mat(4,2),beta_mat(4,3),... 
                 beta_mat(5,1),beta_mat(5,2),beta_mat(5,3),... 
                 x_1(1),x_1(2),x_1(3));
    
end
