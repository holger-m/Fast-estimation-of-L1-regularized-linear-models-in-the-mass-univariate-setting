function [Matlab_b_full, Matlab_b0] = wrapper_matlab(X, Y, lambda_seq)

Matlab_b_full = NaN(size(X,2), size(Y,2), size(lambda_seq,2));
Matlab_b0 = NaN(1, size(Y,2), size(lambda_seq,2));

for k = 1:size(Y,2)
    
    [beta,FitInfo] = lasso(X, Y(:,k), 'Lambda', lambda_seq, 'Standardize', false, 'RelTol', 1e-3);
    
    b0_Matlab = fliplr(FitInfo.Intercept);
    b_Matlab = fliplr(beta);
    
    Matlab_b0(1,k,:) = b0_Matlab;
    Matlab_b_full(:,k,:) = b_Matlab;

end
