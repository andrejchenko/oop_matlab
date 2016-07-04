classdef Neighbours < handle
    properties 
        x_coor
        y_coor
        size1
        size2
        neighbours
        neighboursData
        neighMatrix
        testNeighLabels
        neighData
        
        svmNeighbours
        svmNeighboursData
        svmMeighMatrix
        svmTestNeighLabels
        svmNeighData
        
        sunNeighbours
        sunNeighboursData
        sunMeighMatrix
        sunTestNeighLabels
        sunNeighData
    end
    
    methods
        function obj2 = getXNeighbours(obj1, obj2)
            % input: trainPixIndClass,testPixIndClass,numClasses,indian_pines,classTrainIndex,numNeigh
            obj1.size1 = size(obj2.indian_pines,1);
            obj1.size2 = size(obj2.indian_pines,2);
            
            for i = 1:(obj2.numClasses)
                for j = 1: size(obj2.trainPixIndClass{i},1)
                    %neighbours{i}{j} = zeros(4,2); % 4 neighbours
                    obj1.neighbours{i}{j} = []; % 4 neighbours
                    obj1.neighboursData{i}{j} = [];
                    centerPix_idx = obj2.classTrainIndex{i}(j);
                    obj1.x_coor = obj2.trainPixIndClass{i}(j,1);
                    obj1.y_coor = obj2.trainPixIndClass{i}(j,2);
                    inside = obj1.checkXBoundary(); % check if we are inside the boundaries of our indian pines image
                    for z = 1:size(inside,2)
                        if(inside(z) == 1)
                            % Check wether the neighbouring pixel is part of the
                            % Set of test Pixels
                            %obj1.checkTestPixIndexClass(z,obj2);% if yes remove it from the testing pixels
                            obj2.testPixIndClass = obj1.checkTestPixNN(obj1.x_coor,obj1.y_coor,z, obj2.testPixIndClass,obj2.numClasses);
                            % Add it to the neighbouring set of the current
                            % (center) pixel
                            obj1.addXNeighbours(z,i,j,obj2);
                        else
                            %str = 'Some neighbouring pixels are outside the boundaries of the image: image pines'
                            %str
                        end
                    end
                end
            end
        end
        
        function inside = checkXBoundary(obj1)
            inside = [0 0 0 0 0 0 0 0];
            
            if(obj1.x_coor + 1 <= obj1.size1)&&(obj1.y_coor - 1 <= obj1.size2)&&(obj1.x_coor + 1 >= 1)&&(obj1.y_coor - 1 >= 1) % up right corner
                inside(1) = 1;
            end
            
            if(obj1.x_coor + 1 <= obj1.size1)&&(obj1.y_coor <= obj1.size2)&&(obj1.x_coor + 1 >= 1)&&(obj1.y_coor >= 1) % right
                inside(2) = 1;
            end
            
            if(obj1.x_coor + 1 <= obj1.size1)&&(obj1.y_coor + 1<= obj1.size2)&&(obj1.x_coor + 1 >= 1)&&(obj1.y_coor +1 >= 1) % low right corner
                inside(3) = 1;
            end
            
            if(obj1.x_coor <= obj1.size1)&&(obj1.y_coor + 1<= obj1.size2)&&(obj1.x_coor >= 1)&&(obj1.y_coor +1 >= 1) % bottom
                inside(4) = 1;
            end
            
            if(obj1.x_coor-1<= obj1.size1)&&(obj1.y_coor + 1<= obj1.size2)&&(obj1.x_coor-1 >= 1)&&(obj1.y_coor +1 >= 1) % low left
                inside(5) = 1;
            end
            
            if(obj1.x_coor-1<= obj1.size1)&&(obj1.y_coor<= obj1.size2)&&(obj1.x_coor-1 >= 1)&&(obj1.y_coor >= 1) % left
                inside(6) = 1;
            end
            
            if(obj1.x_coor-1<= obj1.size1)&&(obj1.y_coor-1<= obj1.size2)&&(obj1.x_coor-1 >= 1)&&(obj1.y_coor-1 >= 1) % up left
                inside(7) = 1;
            end
            
            if(obj1.x_coor<= obj1.size1)&&(obj1.y_coor-1<= obj1.size2)&&(obj1.x_coor >= 1)&&(obj1.y_coor-1 >= 1) % top
                inside(8) = 1;
            end
        end
        
        function checkTestPixIndexClass(obj1,z,obj2)
            %obj1.x_coor,obj1.y_coor,z,obj2.testPixIndClass,obj2.numClasses
            if(z==1) % up right
                % x_coor + 1
                % y_coor - 1
                remRows =[];
                for i=1:obj2.numClasses
                    for j = 1: size(obj2.testPixIndClass{i},1)
                        x_test = obj2.testPixIndClass{i}(j,1);
                        y_test = obj2.testPixIndClass{i}(j,2);
                        
                        if(obj1.x_coor + 1 == x_test) && (obj1.y_coor - 1 == y_test)
                            remRows = [remRows;j];
                            %testPixIndClass{i}(j,:)=[];
                        end
                    end
                    for z=1:size(remRows,1)
                        obj2.testPixIndClass{i}(remRows(z),:)=[];
                        remRows =[];
                    end
                end
            end
            
            if(z==2) % right
                % x_coor + 1
                % y_coor
                remRows =[];
                for i=1:obj2.numClasses
                    for j = 1: size(obj2.testPixIndClass{i},1)
                        x_test = obj2.testPixIndClass{i}(j,1);
                        y_test = obj2.testPixIndClass{i}(j,2);
                        
                        if(obj1.x_coor + 1 == x_test) && (obj1.y_coor == y_test)
                            remRows = [remRows;j];
                        end
                    end
                    for z=1:size(remRows,1)
                        obj2.testPixIndClass{i}(remRows(z),:)=[];
                        remRows =[];
                    end
                end
            end
            
            if(z==3) % low right
                % x_coor + 1
                % y_coor + 1
                remRows =[];
                for i=1:obj2.numClasses
                    for j = 1: size(obj2.testPixIndClass{i},1)
                        x_test = obj2.testPixIndClass{i}(j,1);
                        y_test = obj2.testPixIndClass{i}(j,2);
                        
                        if(obj1.x_coor + 1 == x_test) && (obj1.y_coor + 1 == y_test)
                            remRows = [remRows;j];
                        end
                    end
                    for z=1:size(remRows,1)
                        obj2.testPixIndClass{i}(remRows(z),:)=[];
                        remRows =[];
                    end
                end
            end
            
            if(z==4) % bottom
                % x_coor
                % y_coor + 1
                remRows =[];
                for i=1:obj2.numClasses
                    for j = 1: size(obj2.testPixIndClass{i},1)
                        x_test = obj2.testPixIndClass{i}(j,1);
                        y_test = obj2.testPixIndClass{i}(j,2);
                        
                        if(obj1.x_coor == x_test) && (obj1.y_coor + 1 == y_test)
                            remRows = [remRows;j];
                        end
                    end
                    for z=1:size(remRows,1)
                        obj2.testPixIndClass{i}(remRows(z),:)=[];
                        remRows =[];
                    end
                end
            end
            
            if(z==5) % low left
                % x_coor -1
                % y_coor + 1
                remRows =[];
                for i=1:obj2.numClasses
                    for j = 1: size(obj2.testPixIndClass{i},1)
                        x_test = obj2.testPixIndClass{i}(j,1);
                        y_test = obj2.testPixIndClass{i}(j,2);
                        
                        if(obj1.x_coor - 1 == x_test) && (obj1.y_coor + 1 == y_test)
                            remRows = [remRows;j];
                        end
                    end
                    for z=1:size(remRows,1)
                        obj2.testPixIndClass{i}(remRows(z),:)=[];
                        remRows =[];
                    end
                end
            end
            
            if(z==6) % left
                % x_coor -1
                % y_coor
                remRows =[];
                for i=1:obj2.numClasses
                    for j = 1: size(obj2.testPixIndClass{i},1)
                        x_test = obj2.testPixIndClass{i}(j,1);
                        y_test = obj2.testPixIndClass{i}(j,2);
                        
                        if(obj1.x_coor - 1 == x_test) && (obj1.y_coor == y_test)
                            remRows = [remRows;j];
                        end
                    end
                    for z=1:size(remRows,1)
                        obj2.testPixIndClass{i}(remRows(z),:)=[];
                        remRows =[];
                    end
                end
            end
            
            if(z==7) % up left
                % x_coor -1
                % y_coor -1
                remRows =[];
                for i=1:obj2.numClasses
                    for j = 1: size(obj2.testPixIndClass{i},1)
                        x_test = obj2.testPixIndClass{i}(j,1);
                        y_test = obj2.testPixIndClass{i}(j,2);
                        
                        if(obj1.x_coor - 1 == x_test) && (obj1.y_coor - 1 == y_test)
                            remRows = [remRows;j];
                        end
                    end
                    for z=1:size(remRows,1)
                        obj2.testPixIndClass{i}(remRows(z),:)=[];
                        remRows =[];
                    end
                end
            end
            
            if(z==8) % top
                % x_coor
                % y_coor - 1
                remRows =[];
                for i=1:obj2.numClasses
                    for j = 1: size(obj2.testPixIndClass{i},1)
                        x_test = obj2.testPixIndClass{i}(j,1);
                        y_test = obj2.testPixIndClass{i}(j,2);
                        
                        if(obj1.x_coor == x_test) && (obj1.y_coor - 1 == y_test)
                            remRows = [remRows;j];
                        end
                    end
                    for z=1:size(remRows,1)
                        obj2.testPixIndClass{i}(remRows(z),:)=[];
                        remRows =[];
                    end
                end
            end
        end

        function addXNeighbours(obj1,z,i,j,obj2)
            
            if(obj2.numNeigh == 4)
                if(z == 1) % up right corner
                    obj1.neighbours{i}{j} =[obj1.neighbours{i}{j}; obj1.x_coor + 1 obj1.y_coor - 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor + 1,obj1.y_coor - 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(1,1) = x_coor + 1;
                    %neighbours{i}{j}(1,2) = y_coor - 1;
                elseif(z==3) % low right corner
                    obj1.neighbours{i}{j} =[ obj1.neighbours{i}{j}; obj1.x_coor + 1 obj1.y_coor + 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor + 1,obj1.y_coor + 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(2,1) = x_coor + 1;
                    %neighbours{i}{j}(2,2) = y_coor + 1;
                elseif(z == 5) % low left corner
                    obj1.neighbours{i}{j} =[ obj1.neighbours{i}{j}; obj1.x_coor - 1 obj1.y_coor + 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor - 1,obj1.y_coor + 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(3,1) = x_coor - 1;
                    %neighbours{i}{j}(3,2) = y_coor + 1;
                elseif(z == 7) % up left
                    obj1.neighbours{i}{j} =[ obj1.neighbours{i}{j}; obj1.x_coor - 1 obj1.y_coor - 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor - 1,obj1.y_coor - 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(4,1) = x_coor - 1;
                    %neighbours{i}{j}(4,2) = y_coor - 1;
                end
            elseif(obj2.numNeigh == 8)
                if(z == 1) % up right corner
                    obj1.neighbours{i}{j} =[obj1.neighbours{i}{j}; obj1.x_coor + 1 obj1.y_coor - 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor + 1,obj1.y_coor - 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(1,1) = x_coor + 1;
                    %neighbours{i}{j}(1,2) = y_coor - 1;
                elseif(z == 2) % right
                    obj1.neighbours{i}{j} =[obj1.neighbours{i}{j}; obj1.x_coor + 1 obj1.y_coor];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor + 1,obj1.y_coor,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(1,1) = x_coor + 1;
                    %neighbours{i}{j}(1,2) = y_coor;
                elseif(z==3) % low right corner
                    obj1.neighbours{i}{j} =[ obj1.neighbours{i}{j}; obj1.x_coor + 1 obj1.y_coor + 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor + 1,obj1.y_coor + 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(2,1) = x_coor + 1;
                    %neighbours{i}{j}(2,2) = y_coor + 1;
                elseif(z==4) % bottom
                    obj1.neighbours{i}{j} =[ obj1.neighbours{i}{j}; obj1.x_coor obj1.y_coor + 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor,obj1.y_coor + 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(2,1) = x_coor;
                    %neighbours{i}{j}(2,2) = y_coor + 1;
                elseif(z == 5) % low left corner
                    obj1.neighbours{i}{j} =[ obj1.neighbours{i}{j}; obj1.x_coor - 1 obj1.y_coor + 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor - 1,obj1.y_coor + 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(3,1) = x_coor - 1;
                    %neighbours{i}{j}(3,2) = y_coor + 1;
                elseif(z == 6) % left
                    obj1.neighbours{i}{j} =[ obj1.neighbours{i}{j}; obj1.x_coor - 1 obj1.y_coor];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor - 1,obj1.y_coor,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(3,1) = x_coor - 1;
                    %neighbours{i}{j}(3,2) = y_coor;
                elseif(z == 7) % up left
                    obj1.neighbours{i}{j} =[ obj1.neighbours{i}{j}; obj1.x_coor - 1 obj1.y_coor - 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor - 1,obj1.y_coor - 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(4,1) = x_coor - 1;
                    %neighbours{i}{j}(4,2) = y_coor - 1;
                elseif(z == 8) % top
                    obj1.neighbours{i}{j} =[ obj1.neighbours{i}{j}; obj1.x_coor obj1.y_coor - 1];
                    neigh_Pix = obj2.indian_pines(obj1.x_coor,obj1.y_coor - 1,:);
                    neigh_Pix = reshape(neigh_Pix, 1,size(neigh_Pix,3));
                    obj1.neighboursData{i}{j} =[obj1.neighboursData{i}{j}; neigh_Pix];
                    %neighbours{i}{j}(4,1) = x_coor;
                    %neighbours{i}{j}(4,2) = y_coor - 1;
                end
            end
        end
        
        
        function testPixIndClass = checkTestPixNN(obj1,x_coor,y_coor,z, testPixIndClass,numClasses)
            if(z==1) % up right
                % x_coor + 1
                % y_coor - 1
                for i=1:numClasses
                    %I = sum(testPixIndClass(:, 1) == (x_coor + 1) & testPixIndClass(:, 2) == (y_coor - 1));
                    %[r c] = find((class1(:, 1) == 74 & class1(:, 2) == 99))
                    %class1(r,:)=[]
                    [r c] = find((testPixIndClass{i}(:, 1) == (x_coor + 1) & testPixIndClass{i}(:, 2) == (y_coor - 1)))
                    if(r >0)
                        testPixIndClass{i}(r,:)=[];
                    end
                end
            end
            
            if(z==2) % right
                % x_coor + 1
                % y_coor
                for i=1:numClasses
                    [r c] = find((testPixIndClass{i}(:, 1) == (x_coor + 1) & testPixIndClass{i}(:, 2) == (y_coor)))
                    if(r >0)
                        testPixIndClass{i}(r,:)=[];
                    end
                end
            end
            
            if(z==3) % low right
                % x_coor + 1
                % y_coor + 1
                for i=1:numClasses
                    [r c] = find((testPixIndClass{i}(:, 1) == (x_coor + 1) & testPixIndClass{i}(:, 2) == (y_coor + 1)))
                    if(r >0)
                        testPixIndClass{i}(r,:)=[];
                    end
                end
            end
            
            if(z==4) % bottom
                % x_coor
                % y_coor + 1
                for i=1:numClasses
                    [r c] = find((testPixIndClass{i}(:, 1) == (x_coor) & testPixIndClass{i}(:, 2) == (y_coor + 1)))
                    if(r >0)
                        testPixIndClass{i}(r,:)=[];
                    end
                end
            end
            
            if(z==5) % low left
                % x_coor - 1
                % y_coor + 1
                for i=1:numClasses
                    [r c] = find((testPixIndClass{i}(:, 1) == (x_coor - 1) & testPixIndClass{i}(:, 2) == (y_coor + 1)))
                    if(r >0)
                        testPixIndClass{i}(r,:)=[];
                    end
                end
            end
            
            if(z==6) % left
                % x_coor -1
                % y_coor
                for i=1:numClasses
                    [r c] = find((testPixIndClass{i}(:, 1) == (x_coor - 1) & testPixIndClass{i}(:, 2) == (y_coor)))
                    if(r >0)
                        testPixIndClass{i}(r,:)=[];
                    end
                end
            end
            
            if(z==7) % up left
                % x_coor -1
                % y_coor -1
                for i=1:numClasses
                    [r c] = find((testPixIndClass{i}(:, 1) == (x_coor - 1) & testPixIndClass{i}(:, 2) == (y_coor - 1)))
                    if(r >0)
                        testPixIndClass{i}(r,:)=[];
                    end
                end
            end
            
            if(z==8) % top
                % x_coor
                % y_coor - 1
                for i=1:numClasses
                    [r c] = find((testPixIndClass{i}(:, 1) == (x_coor) & testPixIndClass{i}(:, 2) == (y_coor - 1)))
                    if(r >0)
                        testPixIndClass{i}(r,:)=[];
                    end
                end
            end
        end
        function spectralAngle(nObj,obj)
            %calculate spectral angle between each training pixel and its neighbour
            %if the train pixel and the neighbour are spectrally very different add that neighbour to the svm training set
            %if the train pixel and the neighbour are spectrally (very) similar add that neighbour to the unmixing training set
            for i = 1:(obj.numClasses)
                
                 n = nObj.neighMatrix{i};
                 tr = obj.trainMatrix{i};
            end
            
            
        end
    end
end