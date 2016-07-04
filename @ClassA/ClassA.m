classdef ClassA < handle
    properties 
    Value
    end
    methods
        function r = roundOff(obj)
            r = obj.Value * 3;
            obj.Value = r;
        end
        function r = multiplyBy(obj,n)
            r = [obj.Value] * n;
            obj.Value = r;
        end
    end
end