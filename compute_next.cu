#include  "cuda_helper.h"

__global__
void iter_kernel_jacobi(double *x_old, double *x_new, int npts_x, int npts_y, double *diff);
__global__
void iter_kernel_gs(double *x_old, double *x_new, int npts_x, int npts_y, double *diff);


void compute_next(
    double *x_old_d, double *x_old_h,
    double *x_new_d, int npts_local_x, int npts_local_y,
    double *diff_d, double *diff_h,
    int iter_method)
{
    // transfer data from host to device
    cudaErrCheck(
        cudaMemcpy(x_old_d, x_old_h, npts_local_x * npts_local_y * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemset(diff_d, 0, sizeof(double)));

    dim3 block(8, 8);
    dim3 grid((npts_local_x - 2 + block.x - 1) / block.x,
              (npts_local_y - 2 + block.y - 1) / block.y);

    if(iter_method == 0)
    {
        iter_kernel_jacobi<<<grid, block>>>(x_old_d, x_new_d, npts_local_x, npts_local_y, diff_d);
    }
    else
    {
        iter_kernel_gs<<<grid, block>>>(x_old_d, x_new_d, npts_local_x, npts_local_y, diff_d);
    }

    // transfer data from device to host
    cudaErrCheck(cudaMemcpy(x_old_h, x_new_d, npts_local_x * npts_local_y * sizeof(double), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(diff_h, diff_d, sizeof(double), cudaMemcpyDeviceToHost));
}