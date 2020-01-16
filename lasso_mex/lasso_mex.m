function [b_values, b_indexes, N_nz, b0] = lasso_mex(X, Y, lambda_seq, options)

if nargin < 3
    
    error('Specify X, Y and a lambda sequence!');
    
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

if nargin == 4
    
    n_iter_max = int32(options.n_iter_max);
    tol_value = options.tol_value;
    buffer_factor = options.buffer_factor;
    cpu_load_factor = options.cpu_load_factor;
    
else
    
    n_iter_max = int32(1e5);
    tol_value = 1e-3;
    buffer_factor = 3;
    cpu_load_factor = 1;
    
end

N_cores_max = maxNumCompThreads;
N_cores_set = int32(min(max(floor(cpu_load_factor*N_cores_max), 1), N_cores_max));

N_i = int32(size(X,1));
N_j = int32(size(X,2));
N_j_y = int32(size(Y,2));

N_lambda = int32(size(lambda_seq,2));

N_nz_max = buffer_factor*N_i;

b0 = mean(Y);

Y = Y - repmat(b0, N_i, 1);

cov_xx = X'*X;
cov_xy = X'*Y;

cov_xx = reshape(cov_xx, N_j*N_j, 1);
cov_xy = reshape(cov_xy, N_j_y*N_j, 1);

b_values_cpp = zeros(N_j_y*N_nz_max,1);
b_indexes_cpp = zeros(N_j_y*N_nz_max,1,'int32');
N_nz_cpp = zeros(N_j_y,1,'int32');

b_values = zeros(N_nz_max, N_j_y, N_lambda);
b_indexes = zeros(N_nz_max, N_j_y, N_lambda,'int32');
N_nz = zeros(N_j_y, N_lambda,'int32');
b0 = repmat(b0, 1, 1, N_lambda);

for lambda_no = 1:N_lambda
    
    lambda_value = lambda_seq(1,lambda_no);
    
    b_values_init = b_values_cpp;
    b_indexes_init = b_indexes_cpp;
    N_nz_init = N_nz_cpp;
    
    [b_values_cpp, b_indexes_cpp, N_nz_cpp, error_flag_max_iter, error_flag_N_nz_max] = lasso_mex_cpp(cov_xx, ...
                                                                                                      cov_xy, ...
                                                                                                      N_i, ...
                                                                                                      N_j, ...
                                                                                                      N_j_y, ...
                                                                                                      lambda_value, ...
                                                                                                      n_iter_max, ...
                                                                                                      tol_value, ...
                                                                                                      b_values_init, ...
                                                                                                      b_indexes_init, ...
                                                                                                      N_nz_init, ...
                                                                                                      N_nz_max, ...
                                                                                                      N_cores_set);
    
    if any(error_flag_N_nz_max)
        
        error('N_nz over maximum, larger buffer is required!');
        
    end
    
    if any(error_flag_max_iter)
        
        warning('Max. iter. reached, no convergence!');
        
    end
    
    b_values(:,:,lambda_no) = reshape(b_values_cpp, N_nz_max, N_j_y);
    b_indexes(:,:,lambda_no) = reshape(b_indexes_cpp, N_nz_max, N_j_y);
    N_nz(:,lambda_no) = N_nz_cpp;
    
end

