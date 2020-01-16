
// compile command windows: mex -R2018a COMPFLAGS='$COMPFLAGS /openmp' lasso_mex_cpp.c
// compile command linux:   mex -R2018a CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp' lasso_mex_cpp.c

#include <omp.h>
#include "math.h"
#include "mex.h"

void coordinate_descend(double *cov_xx, double *cov_xy, int N_i, int N_j, int N_j_y, double lambda_value, int n_iter_max, double tol_value, double *b_values_init, int *b_indexes_init, int *N_nz_init, int N_nz_max, int N_cores_set, double *b_values, int *b_indexes, int *N_nz, bool *error_flag_max_iter, bool *error_flag_N_nz_max, bool *b_binary, double *b_values_clean, int *b_indexes_clean, int *N_nz_clean, int *N_nz_old)
{
    
    int j;
    int k;
    bool full_sweep;
    int iter_no;
    int j_loop;
    double b_new;
    int j_act;
    int j_ind;
    double max_quad_diff;
    
    omp_set_num_threads(N_cores_set);
    
    #pragma omp parallel for private(j, full_sweep, iter_no, j_loop, b_new, j_act, j_ind, max_quad_diff)
    for (k = 0; k < N_j_y; k++) {
        
        error_flag_N_nz_max[k] = false;
        error_flag_max_iter[k] = false;

        for (j = 0; j < N_nz_max; j++) {

            b_values[N_nz_max*k+j] = 0;

            b_indexes[N_nz_max*k+j] = 0;

        }

        for (j = 0; j < N_j; j++) {

            b_binary[N_j*k+j] = false;

        }

        N_nz[k] = N_nz_init[k];

        if (N_nz[k] > 0){

            for (j = 0; j < N_nz[k]; j++) {

                b_values[N_nz_max*k+j] = b_values_init[N_nz_max*k+j];

                j_ind = b_indexes_init[N_nz_max*k+j];

                b_indexes[N_nz_max*k+j] = j_ind;

                b_binary[N_j*k+j_ind] = true;

            }

        }

        full_sweep = true;

        for (iter_no = 0; iter_no < n_iter_max; iter_no++) {

            if (iter_no == (n_iter_max - 1)){

                error_flag_max_iter[k] = true;

            }

            if (full_sweep){

                for (j = 0; j < N_nz_max; j++) {

                    b_values_clean[N_nz_max*k+j] = 0;

                    b_indexes_clean[N_nz_max*k+j] = 0;

                }

                for (j = 0; j < N_j; j++) {

                    b_binary[N_j*k+j] = false;

                }

                N_nz_clean[k] = 0;

                if (N_nz[k] > 0){

                    for (j = 0; j < N_nz[k]; j++) {

                        if (b_values[N_nz_max*k+j] != 0){

                            b_values_clean[N_nz_max*k+N_nz_clean[k]] = b_values[N_nz_max*k+j];
                            b_indexes_clean[N_nz_max*k+N_nz_clean[k]] = b_indexes[N_nz_max*k+j];

                            j_ind = b_indexes[N_nz_max*k+j];

                            b_binary[N_j*k+j_ind] = true;

                            N_nz_clean[k] = N_nz_clean[k] + 1;

                        }

                    }

                }

                for (j = 0; j < N_nz_max; j++) {

                    b_values[N_nz_max*k+j] = b_values_clean[N_nz_max*k+j];

                    b_indexes[N_nz_max*k+j] = b_indexes_clean[N_nz_max*k+j];

                }

                N_nz[k] = N_nz_clean[k];

                N_nz_old[k] = N_nz[k];

                for (j_loop = 0; j_loop < N_j; j_loop++) {

                    if (!b_binary[N_j*k+j_loop]){

                        b_new = 0.0;

                        if (N_nz[k] > 0){

                            for (j = 0; j < N_nz[k]; j++) {

                                j_ind = b_indexes[N_nz_max*k+j];

                                b_new = b_new + cov_xx[N_j*j_ind+j_loop]*b_values[N_nz_max*k+j];

                            }

                        }

                        b_new = (cov_xy[N_j*k+j_loop] - b_new)/N_i;

                        if (fabs(b_new) > lambda_value){

                            b_binary[N_j*k+j_loop] = true;

                            b_indexes[N_nz_max*k+N_nz[k]] = j_loop;

                            if (b_new > 0.0){

                                b_values[N_nz_max*k+N_nz[k]] = b_new - lambda_value;

                            }
                            else{

                                b_values[N_nz_max*k+N_nz[k]] = b_new + lambda_value;

                            }

                            N_nz[k] = N_nz[k] + 1;

                            if (N_nz[k] == N_nz_max){

                                error_flag_N_nz_max[k] = true;
                                
                                break;

                            }

                        }

                    }

                }
                
                if (error_flag_N_nz_max[k]){
                    
                    break;
                    
                }

                if (N_nz[k] == N_nz_old[k]){

                    break;

                }
                else{

                    full_sweep = false;

                }

            }
            else{
                
                max_quad_diff = 0.0;

                for (j_loop = 0; j_loop < N_nz[k]; j_loop++) {

                    j_act = b_indexes[N_nz_max*k+j_loop];

                    b_new = 0.0;

                    for (j = 0; j < N_nz[k]; j++) {

                        j_ind = b_indexes[N_nz_max*k+j];

                        b_new = b_new + cov_xx[N_j*j_ind+j_act]*b_values[N_nz_max*k+j];

                    }

                    b_new = (cov_xy[N_j*k+j_act] - b_new)/N_i + b_values[N_nz_max*k+j_loop];

                    if (fabs(b_new) <= lambda_value){

                        b_new = 0.0;

                    }
                    else{

                        if (b_new > 0.0){

                            b_new = b_new - lambda_value;

                        }
                        else{

                            b_new = b_new + lambda_value;

                        }

                    }
                    
                    max_quad_diff = fmax(max_quad_diff, fabs(b_values[N_nz_max*k+j_loop] - b_new));
                    
                    b_values[N_nz_max*k+j_loop] = b_new;

                }     

                if (max_quad_diff < tol_value){

                    full_sweep = true;

                }

            }

        }
    
    }
    
}


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    
    double *cov_xx;
    double *cov_xy;
    int N_i;
    int N_j;
    int N_j_y;
    double lambda_value;
    int n_iter_max;
    double tol_value;
    double *b_values_init;
    int *b_indexes_init;
    int *N_nz_init;
    int N_nz_max;
    int N_cores_set;
    
    double *b_values;
    int *b_indexes;
    int *N_nz;    
    bool *error_flag_max_iter;
    bool *error_flag_N_nz_max;
    
    bool *b_binary;
    double *b_values_clean;
    int *b_indexes_clean;
    int *N_nz_clean;
    int *N_nz_old;
    
    cov_xx = mxGetDoubles(prhs[0]);    
    cov_xy = mxGetDoubles(prhs[1]);
    N_i = (int)mxGetScalar(prhs[2]);
    N_j = (int)mxGetScalar(prhs[3]);
    N_j_y = (int)mxGetScalar(prhs[4]);
    lambda_value = mxGetScalar(prhs[5]);
    n_iter_max = (int)mxGetScalar(prhs[6]);
    tol_value = mxGetScalar(prhs[7]);
    b_values_init = mxGetDoubles(prhs[8]);
    b_indexes_init = mxGetInt32s(prhs[9]);    
    N_nz_init = mxGetInt32s(prhs[10]);
    N_nz_max = (int)mxGetScalar(prhs[11]);
    N_cores_set = (int)mxGetScalar(prhs[12]);
    
    plhs[0] = mxCreateNumericMatrix((mwSize)(N_j_y*N_nz_max), 1, mxDOUBLE_CLASS, mxREAL);
    b_values = mxGetDoubles(plhs[0]);
    
    plhs[1] = mxCreateNumericMatrix((mwSize)(N_j_y*N_nz_max), 1, mxINT32_CLASS, mxREAL);
    b_indexes = mxGetInt32s(plhs[1]);
    
    plhs[2] = mxCreateNumericMatrix((mwSize)N_j_y, 1, mxINT32_CLASS, mxREAL);
    N_nz = mxGetInt32s(plhs[2]);
    
    plhs[3] = mxCreateLogicalMatrix((mwSize)N_j_y, 1);
    error_flag_max_iter = mxGetLogicals(plhs[3]);
    
    plhs[4] = mxCreateLogicalMatrix((mwSize)N_j_y, 1);
    error_flag_N_nz_max = mxGetLogicals(plhs[4]);
    
    plhs[5] = mxCreateLogicalMatrix((mwSize)(N_j_y*N_j), 1);
    b_binary = mxGetLogicals(plhs[5]);
    
    plhs[6] = mxCreateNumericMatrix((mwSize)(N_j_y*N_nz_max), 1, mxDOUBLE_CLASS, mxREAL);
    b_values_clean = mxGetDoubles(plhs[6]);
    
    plhs[7] = mxCreateNumericMatrix((mwSize)(N_j_y*N_nz_max), 1, mxINT32_CLASS, mxREAL);
    b_indexes_clean = mxGetInt32s(plhs[7]);
    
    plhs[8] = mxCreateNumericMatrix((mwSize)N_j_y, 1, mxINT32_CLASS, mxREAL);
    N_nz_clean = mxGetInt32s(plhs[8]);
    
    plhs[9] = mxCreateNumericMatrix((mwSize)N_j_y, 1, mxINT32_CLASS, mxREAL);
    N_nz_old = mxGetInt32s(plhs[9]);
    
    coordinate_descend(cov_xx, cov_xy, N_i, N_j, N_j_y, lambda_value, n_iter_max, tol_value, b_values_init, b_indexes_init, N_nz_init, N_nz_max, N_cores_set, b_values, b_indexes, N_nz, error_flag_max_iter, error_flag_N_nz_max, b_binary, b_values_clean, b_indexes_clean, N_nz_clean, N_nz_old);
    
}
