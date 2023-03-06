function [b_full] = convert_betas_sparse_to_full(b_values, b_indexes, N_nz, N_j)

b_full = zeros(N_j, size(b_values,2), size(b_values,3));

for k = 1:size(b_values,2)
    
    for j = 1:size(b_values,3)

        b_values_curr = b_values(1:N_nz(k,j),k,j);
        b_indexes_curr = b_indexes(1:N_nz(k,j),k,j) + 1;  % C++ 0-based index

        b_full(b_indexes_curr,k,j) = b_values_curr;

    end

end
