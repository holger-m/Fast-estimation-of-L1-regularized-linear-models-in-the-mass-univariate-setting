
// compile on Windows: mex -R2018a ADMMcublasOverMex.cpp ADMMcublasOver.obj '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64' -lcudart -lcublas

// compile on Linux: mex -R2018a ADMMcublasOverMex.cpp ADMMcublasOver.o -L/usr/local/cuda/lib64 -lcudart -lcublas

#include "mex.h"
#include "ADMMcublasOver.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    int N_i;
    int N_j;
    int N_n;
    int N_batch;
    int n_iter_max;

    N_i = (int)mxGetScalar(prhs[0]);
    N_j = (int)mxGetScalar(prhs[1]);
    N_n = (int)mxGetScalar(prhs[2]);
    N_batch = (int)mxGetScalar(prhs[3]);
    n_iter_max = (int)mxGetScalar(prhs[4]);

    float *z_in_host;
    z_in_host = mxGetSingles(prhs[5]);

    float *u_in_host;
    u_in_host = mxGetSingles(prhs[6]);

    float *lambda_value_in_host;
    lambda_value_in_host = mxGetSingles(prhs[7]);

    float *Atb_active_in_host;
    Atb_active_in_host = mxGetSingles(prhs[8]);

    float *At_LU_inv_in_host;
    At_LU_inv_in_host = mxGetSingles(prhs[9]);

    float *A_in_host;
    A_in_host = mxGetSingles(prhs[10]);

    float *tol_value_in_host;
    tol_value_in_host = mxGetSingles(prhs[11]);
  
    float *z_host_out;
    plhs[0] = mxCreateNumericMatrix((mwSize)N_n, 1, mxSINGLE_CLASS, mxREAL);
    z_host_out = mxGetSingles(plhs[0]);

    float *u_host_out;
    plhs[1] = mxCreateNumericMatrix((mwSize)N_n, 1, mxSINGLE_CLASS, mxREAL);
    u_host_out = mxGetSingles(plhs[1]);

    bool *error_flag_max_iter;
    plhs[2] = mxCreateLogicalMatrix((mwSize)1, 1);
    error_flag_max_iter = mxGetLogicals(plhs[2]);
    
    ADMM_cublas_over(N_i, N_j, N_n, N_batch, n_iter_max, z_in_host, u_in_host, lambda_value_in_host, Atb_active_in_host, At_LU_inv_in_host, A_in_host, tol_value_in_host, z_host_out, u_host_out, error_flag_max_iter);
    
}
