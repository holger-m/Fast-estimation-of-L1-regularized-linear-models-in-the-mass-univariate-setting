function [B, B0] = lasso_gpu(X, Y, lambda_seq, options)

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
    
    n_iter_max = options.n_iter_max;
    tol_value = single(options.tol_value);
    buffer_size = options.buffer_size;
    
else
    
    n_iter_max = 1e5;
    tol_value = single(1e-3);
    buffer_size = 8192;
    
end


%% get sizes

N_i = size(X,1);
N_j = size(X,2);
N_j_y = size(Y,2);
N_lambda = size(lambda_seq,2);

lambda_seq = single(lambda_seq);


%% convert to Boyd specs

A = X;
b = Y;

clear X
clear Y


%% precompute Cholesky etc.

B0 = mean(b);

b = b - repmat(B0, N_i, 1);

Atb = A'*b/N_i;

Atb = single(Atb);

over_flag = N_j > N_i;

if over_flag
    
    At_LU_inv = (A'/(eye(N_i) + (A*A')/N_i))/N_i;
    
    At_LU_inv = single(At_LU_inv);
    
else
    
    LU_inv = inv((A'*A)/N_i + eye(N_j));
    
    LU_inv = single(LU_inv);
    
end

clear L
clear b

A = single(A);


%% initialize GPU 

D = gpuDevice;

reset(D);

if over_flag

    At_LU_inv_gpu = gpuArray(At_LU_inv);

    A_gpu = gpuArray(A);

else

    LU_inv_gpu = gpuArray(LU_inv);

end


%% prepare GPU batches

N_batches = ceil(N_j_y/buffer_size);

batch_size = ceil(N_j_y/N_batches);

batch_grid = false(N_batches, N_j_y);

for batch_no = 1:N_batches
    
    batch_start = batch_size*(batch_no-1)+1;
    
    batch_end = min(batch_size*batch_no, N_j_y);
    
    batch_grid(batch_no, batch_start:batch_end) = true;
    
end

N_batch_sizes_vec = sum(batch_grid,2);


%% loop through batches and lambda sequence

B = zeros(N_j, N_j_y, N_lambda, 'single');
B0 = repmat(single(B0), 1, 1, N_lambda);

for batch_no = 1:N_batches
    
    batch_bin = batch_grid(batch_no,:);
    
    N_batch = N_batch_sizes_vec(batch_no,1);
    
    Atb_batch_gpu  = gpuArray(Atb(:,batch_bin));

    for lambda_no = 1:N_lambda

        lambda_value = lambda_seq(1,lambda_no);

        lambda_value_gpu = gpuArray(lambda_value);

        if lambda_no == 1

            z_gpu = zeros(N_j, N_batch, 'single', 'gpuArray');
            u_gpu = zeros(N_j, N_batch, 'single', 'gpuArray');

        end
        
        
        for iter_no = 1:n_iter_max
            
            q_gpu = Atb_batch_gpu + z_gpu - u_gpu;
            
            if over_flag
                
                x_gpu = q_gpu - At_LU_inv_gpu*(A_gpu*q_gpu);
                
            else
                
                x_gpu = LU_inv_gpu*q_gpu;
                
            end
            
            zold_gpu = z_gpu;
            
            z_gpu = max((x_gpu + u_gpu) - lambda_value_gpu, 0) - max(-(x_gpu + u_gpu) - lambda_value_gpu, 0);
            
            u_gpu = u_gpu + x_gpu - z_gpu;
            
            delta_max_abs_prim = max(max(abs(x_gpu - z_gpu)));
            delta_max_abs_dual = max(max(abs(z_gpu - zold_gpu)));
            
            conv_flag = gather(delta_max_abs_prim) <  tol_value && gather(delta_max_abs_dual) < tol_value;
            
            if conv_flag
                
                B(:,batch_bin,lambda_no) = gather(z_gpu);

                break;

            end
            
            if iter_no == n_iter_max

                warning('Max. iter. reached, no convergence!');

            end
            
        end
        
    end

end

reset(D);
