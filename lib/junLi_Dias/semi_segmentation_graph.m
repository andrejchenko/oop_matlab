function [w,L] = semi_segmentation_graph(x_L_U,class_L,alpha,beta,MMiter,w0)
% 
%
% ----------------- Input Parameters -------------------------------------
%
% x_L_U ->      training set (each column represent a sample).
%           x_L can be samples or functions of samples, i.e., 
%           kernels, basis, etc.
% class_L ->      class (1,2,...,m)
% alpha  -  shape parameter of Gamma distribution, 
%           default setting: alpha = [1e-1 1e-3]
%
% beta   -  inverse scale parameter of Gamma distribution 
%           default setting: beta = [1e-1 1e-3]

% 
% MMiter -> Number of iterations (default 100)

%
% Copyright (Jul, 2009):        José Bioucas-Dias (bioucas@lx.it.pt)
%                               Jun li (jun@lx.it.pt)
%
%  This algorithm is distributed under the terms of  the GNU General 
%  Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%
% More details in:
%
% [1] J. Li, J. Bioucas-Dias, and A. Plaza. Semi-supervised Hyperspectral 
%     classification.  in  First IEEE GRSS Workshop on Hyperspectral Image 
%     and Signal Processing-WHISPERS'2009, Grenoble, France,  2009.

%
% [2] J. Li, J. Bioucas-Dias and A. Plaza. Semi-Supervised Hyperspectral
%     Image Segmentation Using Multinomial Logistic Regression with Active 
%     Learning. IEEE Transactions on Geoscience and Remote Sensing, vol.48, 
%     no. 11, pp. 4085-4098, 2010


if nargin < 6
    MMiter = 100;
end
if nargin < 5
    beta(1) = 1e-1;
    beta(2) = 1e-3;
end

if nargin <4
    alpha(1) = 1e-1;
    alpha(2) = 1e-3;
end
% number of classes 
m = max(class_L);
n = length(class_L);
% % % % % auxiliar matrix to compute a bound for the logistic hessian
U = -1/2*(eye(m-1)-ones(m-1)/m); 
% % % % % convert y into binary information
Y = zeros(m,n);
for i=1:n
    Y(class_L(i),i) = 1;
end
% remove last line
Y=Y(1:m-1,:);

%  labelled samples
x_L = x_L_U(:,1:n);
% build Rx
Rx = x_L*x_L';

[d,N] = size(x_L_U);

x = x_L_U;
for i = 1:N
    for j = 1:N
        beta(i,j) = exp(-norm(x(:,i)-x(:,j),'fro'));
    end
end


beta1 = beta*ones(N,1);

delta1 = diag(beta1) - beta;

Ra = x_L_U*delta1*x_L_U';

% initialize w with ridge regression fitted to w'*x = 10 in the class
% and w'*x = -10 outside the class
if (nargin < 7)
    alpha0 = 1e-5;
    w=(Rx+alpha0*eye(d))\(x_L*(20*Y'-10*ones(size(Y'))));
else
    w=w0;
end
% lambda=0.004;

% Block GS iterative scheme to compute w
for i=1:MMiter
%     fprintf('\n i = %d',i);
    % compute the  multinomial distributions (one per sample)
    aux1 = exp(w'*x_L);
    aux2 = 1+sum(aux1,1);
    p =  aux1./repmat(aux2,m-1,1);
    % compute log-likelihood
    L(i) = sum(  sum(Y.*(w'*x_L),1) -log(aux2));
    % compute derivative
    dg = Rx*w*U'- x_L*(Y-p)';
    % acumulate 
    a = Rx*w*U';
    % bloc GS
     small = 1e-7;
    for t=1:1
      for k=1:m-1
          lamda0 = (2*alpha(1) + d)/(2*beta(1) + (w(:,k)'*Ra*w(:,k)));
          lamda1 = (2*alpha(2) + d)/(2*beta(2) + (w(:,k)'*Ra*w(:,k)));
          Rr = lamda0*Ra + lamda1*eye(d);
          Ak = U(k,k)*Rx - Rr;
          y = dg(:,k) - a(:,k) + Ak*w(:,k);
          w(:,k) = inv(Ak)*y;
      end
    end
    
end

