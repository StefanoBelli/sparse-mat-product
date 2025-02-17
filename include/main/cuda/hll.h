#ifndef HLL_CUDA_H
#define HLL_CUDA_H

#include <cstdint>
#include <main/cuda/dims_type.h>

dims_type get_dims_for_hll_v1(int, const cudaDeviceProp&);
__global__ void __kernel_hll_v1(
        const double **as,
        const uint64_t **ja,
        const uint64_t *maxnzs,
        const size_t *pitches_as,
        const size_t *pitches_ja,
        uint64_t numblks, 
        uint64_t hs, 
        uint64_t m, 
        const double *x, 
        double *y);

#endif