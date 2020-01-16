function [lambda_start] = calculate_lambda_start(X, Y)

if nargin < 2
    
    error('Specify X and Y!');
    
end

if any(abs(mean(X)) > 1e4*eps)
    
    error('X must be z-scored!')
    
end

if any(abs(var(X,1) - 1) > 1e4*eps)
    
    error('X must be z-scored with flag = 1!')
    
end

if size(X,1) ~= size(Y,1)
    
    error('X and Y must have the same number of rows!');
    
end

cov_xy = X'*Y;

lambda_start = max(abs(cov_xy(:)))/size(X,1) - 1e4*eps;
