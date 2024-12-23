#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define cudaErrCheck(call) {                                            \
    const cudaError_t err = call;                                       \
    if (err != cudaSuccess) {                                           \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                   \
        printf("code:%d, reason: %s\n", err, cudaGetErrorString(err));  \
        exit(1);                                             \
    }                                                                   \
}
