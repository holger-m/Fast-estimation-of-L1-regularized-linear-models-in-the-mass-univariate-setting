#ifndef __ADMMCUBLASOVER_H__
#define __ADMMCUBLASOVER_H__

extern void ADMM_cublas_over(int N_i, int N_j, int N_n, int N_batch, int n_iter_max, float *z_in_host, float *u_in_host, float *lambda_value_in_host, float *Atb_active_in_host, float *At_LU_inv_in_host, float *A_in_host, float *tol_value_in_host, float *z_host_out, float *u_host_out, bool *error_flag_max_iter);

#endif 
