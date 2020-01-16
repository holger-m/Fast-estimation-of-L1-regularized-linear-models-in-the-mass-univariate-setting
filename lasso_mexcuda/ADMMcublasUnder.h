#ifndef __ADMMCUBLASUNDER_H__
#define __ADMMCUBLASUNDER_H__

extern void ADMM_cublas_under(int N_j, int N_n, int N_batch, int n_iter_max, float *z_in_host, float *u_in_host, float *lambda_value_in_host, float *Atb_active_in_host, float *LU_inv_in_host, float *tol_value_in_host, float *z_host_out, float *u_host_out, bool *error_flag_max_iter);

#endif 
