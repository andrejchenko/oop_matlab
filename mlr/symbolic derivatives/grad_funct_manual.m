function y = grad_funct_manual(x_1,beta_mat)
    %x_1 = sym('x', [1 3]);
    %beta_mat = sym('b', [5 3]);
    y = x_1(1) - (exp(beta_mat(1,:)*x_1')/(1+sum(exp(beta_mat * x_1'))))*x_1(1);
end

 