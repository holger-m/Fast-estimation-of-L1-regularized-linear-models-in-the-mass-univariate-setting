
// compile: system('nvcc -c ADMMcublasUnder.cu');
        
// alternatively (on Windows): system('nvcc -c ADMMcublasUnder.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"');

#include "ADMMcublasUnder.h"
#include "math.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

__global__ void soft_thres(float *x_out, float *u_out, float *z_out, float const * const lambda_value_in, int N_n)
{
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N_n){
        
        z_out[i] = fmaxf(x_out[i] + u_out[i] - lambda_value_in[0], 0.0f) - fmaxf(-x_out[i] - u_out[i] - lambda_value_in[0], 0.0f);
        
    }

}

__global__ void delta_abs_value(float *delta_abs_out, int N_n)
{
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N_n){
        
        delta_abs_out[i] = fabsf(delta_abs_out[i]);
        
    }

}

__global__ void determine_convergence(float *delta_abs_out, int max_index, float *z_old_out, int max_index_dual, float const * const tol_value_in, bool *conv_bool_out)
{
    
    if (delta_abs_out[max_index-1] < tol_value_in[0] && z_old_out[max_index_dual-1] < tol_value_in[0]){  // 1-based index returned by cublasIsamax
    
        conv_bool_out[0] = true;
        
    }
    else
    {
        
        conv_bool_out[0] = false;
        
    }

}

void ADMM_cublas_under(int N_j, int N_n, int N_batch, int n_iter_max, float *z_in_host, float *u_in_host, float *lambda_value_in_host, float *Atb_active_in_host, float *LU_inv_in_host, float *tol_value_in_host, float *z_host_out, float *u_host_out, bool *error_flag_max_iter)
{
    
    float *dz_out;
    cudaMalloc(&dz_out, N_n*sizeof(float));

    float *du_out;
    cudaMalloc(&du_out, N_n*sizeof(float));

    float *dlambda_value_in;
    cudaMalloc(&dlambda_value_in, sizeof(float));
    
    float *dAtb_active_in;
    cudaMalloc(&dAtb_active_in, N_n*sizeof(float));

    float *dLU_inv_in;
    cudaMalloc(&dLU_inv_in, (N_j*N_j)*sizeof(float));

    float *dtol_value_in;
    cudaMalloc(&dtol_value_in, sizeof(float));

    cudaMemcpy(dz_out, z_in_host, N_n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(du_out, u_in_host, N_n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dlambda_value_in, lambda_value_in_host, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dAtb_active_in, Atb_active_in_host, N_n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dLU_inv_in, LU_inv_in_host, (N_j*N_j)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dtol_value_in, tol_value_in_host, sizeof(float), cudaMemcpyHostToDevice);

    float *ddelta_abs_out;
    cudaMalloc(&ddelta_abs_out, N_n*sizeof(float));

    float *dq_out;
    cudaMalloc(&dq_out, N_n*sizeof(float));

    float *dx_out;
    cudaMalloc(&dx_out, N_n*sizeof(float));

    float *dz_old_out;
    cudaMalloc(&dz_old_out, N_n*sizeof(float));

    bool *dconv_bool_out;
    cudaMalloc(&dconv_bool_out, sizeof(bool));  

    float scalar_p1;
    float scalar_0;
    float scalar_m1;
    int const threadsPerBlock = 256;
    int blocksPerGrid;
    int max_index;
    int max_index_dual;
    bool *conv_bool_host = new bool[1];
    int iter_no;

    scalar_p1 = 1.0f;
    scalar_0 = 0.0f;
    scalar_m1 = -1.0f;

    blocksPerGrid = (N_n + threadsPerBlock - 1) / threadsPerBlock;

    cublasHandle_t handle;

    cublasCreate(&handle); 
    
    for (iter_no = 0; iter_no < n_iter_max; iter_no++) {
        
        // ADMM
        
        cublasScopy(handle, N_n, dz_out, 1, dq_out, 1);

        cublasSaxpy(handle, N_n, &scalar_m1, du_out, 1, dq_out, 1);

        cublasSaxpy(handle, N_n, &scalar_p1, dAtb_active_in, 1, dq_out, 1);

        cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, N_j, N_batch, &scalar_p1, dLU_inv_in, N_j, dq_out, N_j, &scalar_0, dx_out, N_j);

        cublasScopy(handle, N_n, dz_out, 1, dz_old_out, 1);

        soft_thres<<<blocksPerGrid, threadsPerBlock>>>(dx_out, du_out, dz_out, dlambda_value_in, N_n);
    
        cublasSaxpy(handle, N_n, &scalar_p1, dx_out, 1, du_out, 1);

        cublasSaxpy(handle, N_n, &scalar_m1, dz_out, 1, du_out, 1);

        // prim. conv

        cublasScopy(handle, N_n, dx_out, 1, ddelta_abs_out, 1);

        cublasSaxpy(handle, N_n, &scalar_m1, dz_out, 1, ddelta_abs_out, 1);

        delta_abs_value<<<blocksPerGrid, threadsPerBlock>>>(ddelta_abs_out, N_n);

        cublasIsamax(handle, N_n, ddelta_abs_out, 1, &max_index);

        // dual conv
        
        cublasSaxpy(handle, N_n, &scalar_m1, dz_out, 1, dz_old_out, 1);

        delta_abs_value<<<blocksPerGrid, threadsPerBlock>>>(dz_old_out, N_n);

        cublasIsamax(handle, N_n, dz_old_out, 1, &max_index_dual);
        
        // both conv
        
        determine_convergence<<<1,1>>>(ddelta_abs_out, max_index, dz_old_out, max_index_dual, dtol_value_in, dconv_bool_out);

        cudaMemcpy(conv_bool_host, dconv_bool_out, sizeof(bool), cudaMemcpyDeviceToHost);

        if (conv_bool_host[0])
        {
            
            break;
            
        }

        if (iter_no == (n_iter_max - 1)){

            error_flag_max_iter[0] = true;

        }
        
    }

    cudaMemcpy(z_host_out, dz_out, N_n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_host_out, du_out, N_n*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dz_out);
    cudaFree(du_out);
    cudaFree(ddelta_abs_out);
    cudaFree(dq_out);
    cudaFree(dx_out);
    cudaFree(dz_old_out);
    cudaFree(dconv_bool_out);

    cudaFree(dlambda_value_in);
    cudaFree(dAtb_active_in);
    cudaFree(dLU_inv_in);
    cudaFree(dtol_value_in);
    
}
