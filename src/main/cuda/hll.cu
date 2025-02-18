#include <main/cuda/hll.h>

#define at_pitched_matrix(ty, pitch, base, i, j)  ((ty*)((char*)base + i * pitch) + j)

dims_type get_dims_for_hll_v1(int numblocks, const cudaDeviceProp& device_props) {

    int max_thr_per_blk = device_props.maxThreadsPerBlock;

    if(numblocks <= max_thr_per_blk) {
        dim3 grid_dim(1, 1);
        dim3 block_dim(numblocks, 1);
        return std::make_tuple<>(grid_dim, block_dim, 0);
    }

    double splsz = numblocks / max_thr_per_blk;
    int xgridsz = numblocks % max_thr_per_blk ? std::ceil(splsz) + 1 : splsz;

    dim3 grid_dim(xgridsz, 1);
    dim3 block_dim(max_thr_per_blk, 1);
    return std::make_tuple<>(grid_dim, block_dim, 0);
}

#include <stdio.h>

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
        double *y) {
    
    const int thread_global_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(thread_global_index >= numblks) {
        return;
    }

    const double *my_block_as = as[thread_global_index];
    const size_t my_block_as_pitch = pitches_as[thread_global_index];
    const uint64_t *my_block_ja = ja[thread_global_index];
    const size_t my_block_ja_pitch = pitches_ja[thread_global_index];
    const uint64_t my_block_maxnz = maxnzs[thread_global_index];

    // TODO change here 
    double t[32];

    for(uint64_t i = 0; i < hs; i++) {
        double ell_tmp = 0;
        for(uint64_t j = 0; j < my_block_maxnz; j++) {
            ell_tmp += 
                *at_pitched_matrix(double, my_block_as_pitch, my_block_as, j, i) *
                x[*at_pitched_matrix(uint64_t, my_block_ja_pitch, my_block_ja, j, i)];
            //printf("%ld\n",*at_pitched_matrix(uint64_t, my_block_ja_pitch, my_block_ja, j, i));
        }
        t[i] = ell_tmp;
    }

    uint64_t endrow = hs;
    if (thread_global_index * hs + hs > m) {
        endrow = m - thread_global_index * hs;
    }

    for(uint64_t i = 0; i < endrow; i++) {
        y[i + thread_global_index * hs] = t[i];
    }
}