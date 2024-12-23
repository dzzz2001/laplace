#include "cuda_helper.h"

__inline__ __device__
double warp_reduce(double val)
{
    val += __shfl_xor_sync(0xffffffff, val, 16, 32);
    val += __shfl_xor_sync(0xffffffff, val, 8, 32);
    val += __shfl_xor_sync(0xffffffff, val, 4, 32);
    val += __shfl_xor_sync(0xffffffff, val, 2, 32);
    val += __shfl_xor_sync(0xffffffff, val, 1, 32);
    return val;
}

__global__
void iter_kernel_jacobi(double *x_old, double *x_new, int npts_x, int npts_y, double *diff)
{
    __shared__ double diff_warp[2];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    double diff_local = 0.0;

    if (i > 0 && i < npts_x - 1 && j > 0 && j < npts_y - 1) {
        int idx = i + j * npts_x;
        x_new[idx] = 0.25 * (x_old[idx - 1] + x_old[idx + 1] + x_old[idx - npts_x] + x_old[idx + npts_x]);
        diff_local = (x_new[idx] - x_old[idx]) * (x_new[idx] - x_old[idx]);
    }

    diff_local = warp_reduce(diff_local);

    if(lane_id == 0)
    {
        diff_warp[warp_id] = diff_local;
    }
    __syncthreads();

    if(tid == 0)
    {
        diff_local = diff_warp[0] + diff_warp[1];
        atomicAdd(diff, diff_local);
    }
}

__global__
void iter_kernel_gs(double *x_old, double *x_new, int npts_x, int npts_y, double *diff, bool is_red)
{
    __shared__ double diff_warp[2];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    double diff_local = 0.0;
    
    if(is_red)
    {   
        if (i > 0 && i < npts_x - 1 && j > 0 && j < npts_y - 1) {
            int idx = i + j * npts_x;
            if ((i + j) % 2 == 0) {
                x_new[idx] = 0.25 * (x_old[idx - 1] + x_old[idx + 1] + x_old[idx - npts_x] + x_old[idx + npts_x]);
                diff_local = (x_new[idx] - x_old[idx]) * (x_new[idx] - x_old[idx]);
            }
        }
    }
    else
    {
        if (i > 0 && i < npts_x - 1 && j > 0 && j < npts_y - 1) {
            int idx = i + j * npts_x;
            if ((i + j) % 2 == 1) {
                x_new[idx] = 0.25 * (x_new[idx - 1] + x_new[idx + 1] + x_new[idx - npts_x] + x_new[idx + npts_x]);
                diff_local = (x_new[idx] - x_old[idx]) * (x_new[idx] - x_old[idx]);
            }
        }
    }

    diff_local = warp_reduce(diff_local);

    if(lane_id == 0)
    {
        diff_warp[warp_id] = diff_local;
    }
    __syncthreads();

    if(tid == 0)
    {
        diff_local = diff_warp[0] + diff_warp[1];
        atomicAdd(diff, diff_local);
    }
}
