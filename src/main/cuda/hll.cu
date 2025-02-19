#include <main/cuda/hll.h>

extern "C" {
#include <utils.h>
}

#define at_pitched_matrix(ty, pitch, base, i, j)  ((ty*)((char*)base + (i * pitch)) + j)

dims_type get_dims_for_hll_v1(int numblocks, const cudaDeviceProp& device_props) {

    int max_thr_per_blk = device_props.maxThreadsPerBlock;

    if(numblocks <= max_thr_per_blk) {
        dim3 grid_dim(1, 1);
        dim3 block_dim(numblocks, 1);
        return std::make_tuple<>(grid_dim, block_dim, 0);
    }

    int xgridsz = ceiling_div(numblocks, max_thr_per_blk);
    dim3 grid_dim(xgridsz, 1);
    dim3 block_dim(max_thr_per_blk, 1);
    return std::make_tuple<>(grid_dim, block_dim, 0);
}

// hs must be 32
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

    double t[32];

    for(uint64_t r = 0; r < hs; r++) {
        double ell_tmp = 0;
        for(uint64_t c = 0; c < my_block_maxnz; c++) {
            ell_tmp += 
                *at_pitched_matrix(double, my_block_as_pitch, my_block_as, c, r) *
                x[*at_pitched_matrix(uint64_t, my_block_ja_pitch, my_block_ja, c, r)];
        }

        t[r] = ell_tmp;
    }

    uint64_t endrow = hs;
    if (thread_global_index * hs + hs > m) {
        endrow = m - thread_global_index * hs;
    }

    for(uint64_t i = 0; i < endrow; i++) {
        y[i + thread_global_index * hs] = t[i];
    }
}

dims_type get_dims_for_hll_v2(int numblocks, const cudaDeviceProp& device_props) {
    int max_thrs_per_block = device_props.maxThreadsPerBlock;
    int warp_size = device_props.warpSize;
    int max_warps_each_block = max_thrs_per_block / warp_size;

    if(numblocks <= max_warps_each_block) {
        size_t shmem_size = numblocks * warp_size * sizeof(double);
        dim3 grid_dim(1,1);
        dim3 block_dim(numblocks * warp_size, 1);
        return std::make_tuple<>(grid_dim, block_dim, shmem_size);
    }

    int xgridsz = ceiling_div(numblocks, max_warps_each_block);
    size_t shmem_size = max_thrs_per_block * sizeof(double);
    dim3 grid_dim(xgridsz,1);
    dim3 block_dim(max_thrs_per_block,1);
    return std::make_tuple<>(grid_dim, block_dim, shmem_size);
}

__global__ void __kernel_hll_v2(
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

    extern __shared__ double t_shmem[];
    
    const int warp_global_index = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int thread_idx_in_warp = threadIdx.x % warpSize;

    if(warp_global_index >= numblks || thread_idx_in_warp >= hs) {
        return;
    }
    
    const int warp_local_index = threadIdx.x / warpSize;

    const double *my_block_as = as[warp_global_index];
    const size_t my_block_as_pitch = pitches_as[warp_global_index];
    const uint64_t *my_block_ja = ja[warp_global_index];
    const size_t my_block_ja_pitch = pitches_ja[warp_global_index];
    const uint64_t my_block_maxnz = maxnzs[warp_global_index];

    double ell_tmp = 0;
    for (uint64_t c = 0; c < my_block_maxnz; c++) {
        ell_tmp +=
            *at_pitched_matrix(double, my_block_as_pitch, my_block_as, c, thread_idx_in_warp) *
            x[*at_pitched_matrix(uint64_t, my_block_ja_pitch, my_block_ja, c, thread_idx_in_warp)];
    }

    t_shmem[(warp_local_index * warpSize) + thread_idx_in_warp] = ell_tmp;

    __syncthreads();

    if (thread_idx_in_warp == 0) {
        uint64_t endrow = hs;
        if (warp_global_index * hs + hs > m) {
            endrow = m - warp_global_index * hs;
        }

        for (uint64_t i = 0; i < endrow; i++) {
            y[i + warp_global_index * hs] = t_shmem[(warp_local_index * warpSize) + i];
        }
    }
}