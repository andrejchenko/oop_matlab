%  function demo1_simulated_data()

%--------------Brief description------------------------------------------- 
% 
% This demo correponding to A.1 in [2]: Impact of including a spatial prior 
%
%   
% In summary:
%
%   1- Based on a training set containing  spectral vectors and the
%      respective labels, learn the regressors of a multinomial 
%      logistic regression (MLR) GEM algorithm (see Algorithm 1 in [2])
%
%    2- Based on a multi-level logistic (MLL) prior, to model the 
%       contextual spatial information of the hyperspectral images, 
%       compute the MAP segmentation  via the \alpha-Expanasion 
%       Graph-cut algorithm.
%
%
% -------------------------- Input parameters -----------------------------
%
% -------------------------Algorithm parameters----------------------------
%
% alpha  -  shape parameter of Gamma distribution, 
%           default setting: alpha = [1e-1 1e-3]
%
% beta   -  inverse scale parameter of Gamma distribution 
%           default setting: beta = [1e-1 1e-3]
%
% mu      - the spatial prior regularization  parameter, controlling 
%           the degree of spatial smothness. Tune this parameter to 
%           obtain a good segmentation results.
%
% --------------------hyperpsectral data parameters------------------------
%
% no_classes        - number of classes 
% dim               - dimentionality of the hyperspectral data
% n                 - number of pixels in the image: n = dim*dim
% d                 - number of bands. 
% sigma             - noise standard deviation (std) of dataset
%
% ---------------------------- demo  parameters ---------------------------
%
% GC_toolbox        - this variable takes values in  {Bagon, Bioucas}
%                     default = Bagon.
%                     Graph cut toolbox to implement the alpha-Expansion
%                     algorithm. Both toolboxes are compiled for 
%                     Matlab R2009.
%                     If you face problems, compile the Bagon toolbox in 
%                     you computer by running compile_gc.m. See details in
%                     GCmex1.2
%
% hyperspectral_image - this variable takes values in 
%                       {new_image, existing_image}.   
%                       default = existing_image. 
%                      'new_image' generates a new class label image
%                       according to the MLL distribution.
%                       'existing_image' uses an existing binary image:
%                        demo1_classes.
%                       
% size_train_set    - size of labeled samples
%
% MMiter            - number of Monte Carlo runs to compute the overall
%                     accuracy
%
%
% --------------------------- output parameters ---------------------------
%
% OA_class      - classification accuracies with the GEM algorithm 
%                 (Algorithm 1) in [2]
% OA_seg        - segmentation accuracies with the segmentation algorithm
%                 (Algorithm 2) in [2]
%      
%  
% ------------------------ Example1  --------------------------------------
%
% dim = 128;             
% n   = dim^2;           
% no_classes=2;          % binary problem
% d = 50;                
% size_train_set = 1000;   
% sigma  = sqrt(2);       
% mu=4;                  
% MMiter = 10;       (smaller values  ==> faster simulations and
%                     higher uncertainties in the overall accuracies (OAs))
%   
% -------------------------------------------------------------------------

% Author: Jun Li
%         Jose M. Bioucas-Dias, Sep. 2010
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
%

clear all
close all
clc


%%
%--------------------------------------------------------------------------
%                          data parameter
%--------------------------------------------------------------------------
% set GC toolbox
GC_toolbox = 'Bagon';
% GC_toolbox = 'Bioucas';

% set hyperspectral image
hyperspectral_image = 'existing_image';
% hyperspectral_image = 'new_image';

dim = 128;             % dimentionality
n   = dim^2;           % number of samples
no_classes=2;          % number of classes
d = 50;                % the number of bands

%%
%--------------------------------------------------------------------------
%                     generate an MLL image
%--------------------------------------------------------------------------
if strcmp(hyperspectral_image,'new_image')
% generate a new image
 classes = mll(dim,2,no_classes,2);
else
% using a existing image
load demo1_classes;
end

size_train_set = 100;   % size of labeled samples
sigma  = sqrt(2);       % noise standard deviation (std)
mu=2;                   % MLL soomthness parameter
MMiter = 10;            % number of simulations

% esimate the probability of each class
p1 = sum(classes(:) == 1)/dim/dim;
p2 = sum(classes(:) == 2)/dim/dim;

p0 = 0.5; %p(y_i) = 1/K, assuming without loss of generality.

truth_image = classes;

% buid two antipodal unit vectors
M = zeros(d,2);
M(1:d,1) = 1/sqrt(d);
M(1:d,2) = -1/sqrt(d);

% -------------------------------------------------------------------------
%        generate data sets with noise std sigma
%--------------------------------------------------------------------------
% compute the optilmal ML threshold

lambda0  = sigma^2/2*log(p2/p1);

% compute the theoretical probability of error
pe = erfc((lambda0+1)/sqrt(2)/sigma)*p2/2+erfc((1-lambda0)/sqrt(2)/sigma)*p2/2;
% theoretical OA 
pc = (1-pe)*100;

% generate the data set
x = M(:,classes(:)) + sigma*randn(d,n);

%%
%--------------------------------------------------------------------------
%                         Start MC runs
%--------------------------------------------------------------------------
fprintf('\nStarting Monte Carlo runs \n\n');
for mmiter = 1:MMiter 
    fprintf('MC run %d \n', mmiter);
    % % % generate the  training set
    index1 = find(classes == 1);
    per_index1 = randperm(length(index1));
    index2 = find(classes == 2);
    per_index2 = randperm(length(index2));
    indexes = [index1(per_index1(1:round(size_train_set*p0)))' ...
              index2(per_index2(1:round(size_train_set*(1-p0))))'];
    classes_L = classes(indexes);   % labels in the training set
    x_L = [x(:,indexes)];           %  vectors in the training set

    % % tarining set dimension
    [d,n_L]=size(x_L);
    K = [ones(1,n_L); x_L];

    %---------------------------------------------------------------------
    %                     learn MLR coefficients
    %---------------------------------------------------------------------
    [w,L] = semi_segmentation_graph(K,classes_L);
        
    % normalization
    w = 2*w./repmat(sqrt(sum(w.^2)),d+1,1);
    
    % compute the class probablities
    x_c=x;
    K = [ones(1,n); x_c];
    p=mlogistic(w,K);
    [maxp,class_hat] = max(p);

    %%
    %----------------------------------------------------------------------
    %                    evaluation of the classification algorithm 
    %----------------------------------------------------------------------

    class_map = reshape(class_hat,dim,dim);

    % classification overall accuracy
    OA_class(mmiter) =  sum(class_hat==classes(:)')/n;

    if strcmp(GC_toolbox,'Bagon')
    %----------------------------------------------------------------------
    %               graphcut by Matlab wrapper; 
    %        http://www.wisdom.weizmann.ac.il/~bagon/matlab.html          
    %----------------------------------------------------------------------
  
    % energy for each class
    Dc = reshape((log(p+eps))',[dim dim 2]);
    
    % smoothness term:
    % constant part
    Sc = ones(2) - eye(2);

    gch = GraphCut('open', -Dc, mu*Sc);
    [gch seg] = GraphCut('expand',gch);
    gch = GraphCut('close', gch);
    
    else
        
    %% 
    %----------------------------------------------------------------------
    %        graphcut by another toolbox; Jose Bioucas Dias        
    %----------------------------------------------------------------------
    % energy for each class
    e=-log(p+eps);% log(p12);
    for i=1:no_classes
        eLabel(:,:,i)=reshape(e(i,:),dim,dim);
    end
    % cliques definition
%     cliques = [ 0  1;
%             1  1;
%             1  0;
%             -1  1];
        % % % % % %
    cliques = [ 0  1;
            1  0];
     % % % % % % %
    % % % % % % % labels are: 0, 1, ...
 
    [cliquesm,cliquesn] = size(cliques);
    [em,en,nrLabels] = size(eLabel);
    labels = [1:nrLabels];
    labels = labels-1;
    % Expantion Algorithm
    initLabelling = ones(dim,dim);
    seg = alfaExpansion(eLabel,labels,initLabelling,mu,cliques);
    
    end
        

    %%
    %----------------------------------------------------------------------
    %         evaluation of the segmentation algorithm 
    %----------------------------------------------------------------------
       
%     % plot the segmentation map
%     figure(3), imagesc(seg)      
 
    % segmentation overall accuracy
    OA_seg(mmiter) =  sum(seg(:)+1==classes(:))/n;
    
end

%-------------------------------------------------------------------------%
%        evaluation of the algorithm performance
%-------------------------------------------------------------------------%

% the training set size
fprintf('The final number of  training samples: L =  %d \n', size_train_set);


% compute the mean OAs over MMiter MC runs
mean_classification = mean(OA_class);
fprintf('The classification OA =  %2.2f  \n', mean_classification);
mean_segmentation = mean(OA_seg);
fprintf('The segmemtation OA  =  %2.2f \n', mean_segmentation);


%--------------------------------------------------------------------------
%                    plot the images
%--------------------------------------------------------------------------
%
% ground truth image
%
figure(1),imagesc(truth_image);
title('\bf{Ground truth map}','Fontsize', 14)

% classification map
figure(2), imagesc(class_map)
st = sprintf('classification map: OA = %2.2f', mean_classification);
title(st,'Fontsize', 12)

% segmentation map
figure(3), imagesc(seg)
st = sprintf('segmentation map: OA = %2.2f', mean_segmentation);
title(st,'Fontsize', 12)

% %-----------------------------------------------------------------------%
