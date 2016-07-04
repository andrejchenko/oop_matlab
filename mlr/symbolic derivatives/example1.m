%beta = sym('b', [1 3]);
x_1 = sym('x', [1 3]);

%beta_mat = sym('b_l', [5 3]);
beta_mat = sym('b', [5 3]);

%expr = log(exp(beta*x_1')/(1+sum(exp(beta_mat * x_1'))));
expr = log(exp(beta_mat(1,:)*x_1')/(1+sum(exp(beta_mat * x_1'))))

%grad = gradient(expr, beta);
%grad_1 = gradient(expr, beta(1));


grad = gradient(expr, beta_mat(1,1)); % w.r.t. b_1_1
