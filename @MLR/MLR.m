classdef MLR < handle
    properties
        mlrModel
        w
    end
    methods
        function trainMLR(mObj,obj)
            trainLabels = categorical(obj.trainLabels);
            [B,dev,stats] = mnrfit(obj.trainData',trainLabels'); % trainData should be 80 x 164 format
            mlrModel = B;
            mObj.mlrModel = B;
            save mlrModel mlrModel
        end
        
        function predictMLR(mObj,obj)
            predictions = zeros(size(obj.testData,1),obj.numClasses);
            for i = 1: size(obj.testData,1)
                 %x = [6.2, 3.7, 5.8, 0.2];
                 x = obj.testData(i,:);
                 pihat = mnrval(mObj.mlrModel,x);
                 predictions(i) = pihat;
            end
            
        end
        
        %Code used and adapted from: Jun Li and Jose M. Bioucas-Dias, Sep. 2010
        %demo1_simulated_data.m
        %semi_segmentation_graph.m
        %mlogistic.m
        function trainMLR_Sub(mObj,obj) 
            classes_L = obj.trainLabels;
            % % tarining set dimension
            x_L = obj.trainData'; % so its d x n_l
            [d,n_L]=size(x_L);
            K = [ones(1,n_L); x_L];
        %---------------------------------------------------------------------
        %                     learn MLR coefficients
        %---------------------------------------------------------------------
            [w,L] = semi_segmentation_graph(K,classes_L);
            % normalization
            w = 2*w./repmat(sqrt(sum(w.^2)),d+1,1);
            mObj.w = w;
        end
        
        function [acc,o_acc] = predictMLR_Sub(mObj,obj)
            % compute the class probablities, but in my case only for the
            % test data, not the whole image data
            %x_c=x;
            x_c = obj.testData';
            [d_n,n] = size(x_c);
            K = [ones(1,n); x_c];
            p = mlogistic(mObj.w,K);
            [maxp,class_hat] = max(p);
            
            acc = obj.calcXAccuracy(obj.testLabels,class_hat');
            acc = acc(1)*100;
            o_acc = sum(class_hat==obj.testLabels(:)')/size(obj.testData,1);
            o_acc = o_acc * 100;
        end
    end
end
