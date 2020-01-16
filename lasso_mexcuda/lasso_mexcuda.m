function [B, B0] = lasso_mexcuda(X, Y, lambda_seq, options)

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
    tol_value = single(options.tol_value);
    buffer_size = single(options.buffer_size);
        
else
    
    n_iter_max = int32(1e5);
    tol_value = single(1e-3);
    buffer_size = single(8192);
        
end


%% get sizes

N_i = int32(size(X,1));
N_j = int32(size(X,2));
N_j_y = int32(size(Y,2));
N_lambda = int32(size(lambda_seq,2));

lambda_seq = single(lambda_seq);


%% convert to Boyd specs

boyd_factor = sqrt(size(X,1));

A = X/boyd_factor;
b = Y;

clear X
clear Y

boyd_factor = single(boyd_factor);


%% precompute Cholesky etc.

B0 = mean(b);

b = b - repmat(B0, N_i, 1);

Atb = A'*b;

Atb = single(Atb);

over_flag = N_j > N_i;

if over_flag
    
    L = chol(eye(N_i) + (A*A'), 'lower');
    
    At_LU_inv = A'/(L*L');
    
    At_LU_inv = reshape(At_LU_inv, N_i*N_j, 1);
    
    At_LU_inv = single(At_LU_inv);
    
else
    
    L = chol(A'*A + eye(N_j), 'lower');
    
    LU_inv = inv(L*L');
    
    LU_inv = reshape(LU_inv, N_j*N_j, 1);
    
    LU_inv = single(LU_inv);
    
end

clear L
clear b

A = reshape(A, N_i*N_j, 1);
A = single(A);

%% prepare GPU batches

N_batches = int32(ceil(single(N_j_y)/buffer_size));

batch_size = int32(ceil(single(N_j_y)/single(N_batches)));

batch_grid = false(N_batches, N_j_y);

for batch_no = 1:N_batches
    
    batch_start = batch_size*(batch_no-1)+1;
    
    batch_end = min(batch_size*batch_no, N_j_y);
    
    batch_grid(batch_no, batch_start:batch_end) = true;
    
end

N_batch_sizes_vec = int32(sum(batch_grid,2));


%% loop through batches and lambda sequence

B = zeros(N_j, N_j_y, N_lambda, 'single');
B0 = repmat(single(B0), 1, 1, N_lambda);

for batch_no = 1:N_batches
    
    batch_bin = batch_grid(batch_no,:);
    
    N_batch = N_batch_sizes_vec(batch_no,1);
    
    N_n = N_j*N_batch;
    
    Atb_batch = Atb(:,batch_bin);
    
    Atb_batch = reshape(Atb_batch, N_n, 1);

    for lambda_no = 1:N_lambda

        lambda_value = boyd_factor*lambda_seq(1,lambda_no);

        if lambda_no == 1

            z = zeros(N_n, 1, 'single');
            u = zeros(N_n, 1, 'single');

        end

        if over_flag
            
            [z, u, error_flag_max_iter] = ADMMcublasOverMex(N_i, ... 
                                                            N_j, ... 
                                                            N_n, ... 
                                                            N_batch, ... 
                                                            n_iter_max, ... 
                                                            z, ... 
                                                            u, ... 
                                                            lambda_value, ... 
                                                            Atb_batch, ... 
                                                            At_LU_inv, ... 
                                                            A, ... 
                                                            tol_value);
        
        else
            
            [z, u, error_flag_max_iter] = ADMMcublasUnderMex(N_j, ... 
                                                             N_n, ... 
                                                             N_batch, ... 
                                                             n_iter_max, ... 
                                                             z, ... 
                                                             u, ... 
                                                             lambda_value, ... 
                                                             Atb_batch, ... 
                                                             LU_inv, ... 
                                                             tol_value);
        
        end

        if error_flag_max_iter

            warning('Max. iter. reached, no convergence!');

        end
        
        B(:,batch_bin,lambda_no) = reshape(z, N_j, N_batch);

    end

end

B = B/boyd_factor;
