%% Create random data

X = randn(300,5000);
Y = randn(300,16);

X = (X - repmat(mean(X),size(X,1),1))./repmat(std(X,1),size(X,1),1);  % X must be z-scored!

lambda_seq = 2.^(-2:-1:-6);  % lambda-values must be decreasing to benefit from warm starts


%% Run Matlab native if Statistics and Machine Learning Toolbox exists

ver_out = ver;

if any(strcmp({ver_out.Name}, 'Statistics and Machine Learning Toolbox'))
    
    statistics_flag = true;
    
else
    
    statistics_flag = false;
    
end

if statistics_flag

    disp(' ');
    disp('Matlab native:');

    tic

    [Matlab_b_full, Matlab_b0] = wrapper_matlab(X, Y, lambda_seq);

    toc

else
    
    disp(' ');
    disp('Statistics and Machine Learning Toolbox not found.');
    
end


%% Run lasso_gpu

% Options can be set in an options struct:

% options.n_iter_max = 1e5;
% options.tol_value = 1e-3;
% options.buffer_size = 8192;

disp(' ');
disp('lasso_gpu:');

tic

[B, B0] = lasso_gpu(X, Y, lambda_seq);
% [B, B0] = lasso_gpu(X, Y, lambda_seq, options);

toc

disp(' ');


%% Plots for a few y columns

for k = [1, 2, size(Y,2)-1, size(Y,2)]

    for lambda_no = 1:size(lambda_seq,2)
        
        if statistics_flag

            data_full = [[Matlab_b0(1,k,lambda_no); Matlab_b_full(:,k,lambda_no)],...
                         [B0(1,k,lambda_no); B(:,k,lambda_no)]];
                 
        else
            
            data_full = [B0(1,k,lambda_no); B(:,k,lambda_no)];
            
        end
        
        data_sparse = data_full(any(data_full ~= 0, 2), :);

        figure();
        bar(data_sparse);
        title(['k = ',num2str(k),', lambda = ',num2str(lambda_seq(1,lambda_no))]);

    end

end
