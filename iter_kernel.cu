#include "cuda_helper.h"

__global__
void iter_kernel_jacobi(double *x_old, double *x_new, int npts_x, int npts_y, double *diff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < npts_x - 1 && j > 0 && j < npts_y - 1) {
        int idx = i + j * npts_x;
        x_new[idx] = 0.25 * (x_old[idx - 1] + x_old[idx + 1] + x_old[idx - npts_x] + x_old[idx + npts_x]);
        // TODO: the reduction of diff can be optimized
        atomicAdd(diff, (x_new[idx] - x_old[idx]) * (x_new[idx] - x_old[idx]));
    }
}

__global__
void iter_kernel_gs(double *x_old, double *x_new, int npts_x, int npts_y, double *diff)
{
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;

    // if (i > 0 && i < npts_x - 1 && j > 0 && j < npts_y - 1) {
    //     int idx = i + j * npts_x;
    //     x_new[idx] = 0.25 * (x_new[idx - 1] + x_old[idx + 1] + x_new[idx - npts_x] + x_old[idx + npts_x]);

    // }
}
