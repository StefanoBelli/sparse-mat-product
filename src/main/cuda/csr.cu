#include <main/cuda/csr.h>

extern "C" {
#include <utils.h>
}

dims_type get_dims_for_csr_v1(
        int nrows, 
        const cudaDeviceProp& device_props) {

    int max_thr_per_blk = device_props.maxThreadsPerBlock;

    if(nrows <= max_thr_per_blk) {
        dim3 grid_dim(1, 1);
        dim3 block_dim(nrows, 1);
        return std::make_tuple<>(grid_dim, block_dim, 0);
    }

    int xgridsz = ceiling_div(nrows, max_thr_per_blk);
    dim3 grid_dim(xgridsz, 1);
    dim3 block_dim(max_thr_per_blk, 1);
    return std::make_tuple<>(grid_dim, block_dim, 0);
}

__global__ void __kernel_csr_v1(
        const uint64_t *irp, 
        const uint64_t *ja, 
        const double *as, 
        uint32_t m, 
        const double *x, 
        double *y) {

    const int thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_global_index >= m) {
        return;
    }

    double t = 0;
    for(int j = irp[thread_global_index]; j < irp[thread_global_index + 1]; j++) {
        t += as[j] * x[ja[j]];
    }

    y[thread_global_index] = t;
}

dims_type get_dims_for_csr_v2(
        int nrows, 
        const cudaDeviceProp& device_props) {

    int warp_size = device_props.warpSize;
    int max_thr_per_blk = device_props.maxThreadsPerBlock;

    if(nrows * warp_size <= max_thr_per_blk) {
        dim3 grid_dim(1,1);
        dim3 block_dim(nrows * warp_size, 1);
        return std::make_tuple<>(grid_dim, block_dim, 0);
    }

    int xgridsz = ceiling_div(nrows * warp_size, max_thr_per_blk);
    dim3 grid_dim(xgridsz, 1);
    dim3 block_dim(max_thr_per_blk, 1);
    return std::make_tuple<>(grid_dim, block_dim, 0);
}

__global__ void __kernel_csr_v2(
        const uint64_t *irp, 
        const uint64_t *ja, 
        const double *as, 
        uint32_t m, 
        const double *x, 
        double *y) {

    if(threadIdx.x % warpSize != 0) {
        return;
    }

    const int warp_global_index = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    
    if(warp_global_index >= m) {
        return;
    }

    double t = 0;

    for(int j = irp[warp_global_index]; j < irp[warp_global_index + 1]; j++) {
        t += as[j] * x[ja[j]];
    }

    y[warp_global_index] = t;
}

dims_type get_dims_for_csr_v3(
        int nrows, 
        const cudaDeviceProp& device_props) {

    auto csrv2dims = get_dims_for_csr_v2(nrows, device_props);
    size_t shmemsize = std::get<1>(csrv2dims).x * sizeof(double); 
    return std::make_tuple<>(std::get<0>(csrv2dims), std::get<1>(csrv2dims), shmemsize);
}

__global__ void __kernel_csr_v3(
        const uint64_t *irp, 
        const uint64_t *ja, 
        const double *as, 
        uint32_t m, 
        const double *x, 
        double *y) {

    extern __shared__ double row_shmem[];

    const int warp_global_index = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if(warp_global_index >= m) {
        return;
    }

    const int irps = irp[warp_global_index];
    const int irpe = irp[warp_global_index + 1];
    const int nj = irpe - irps;

    const int thread_idx_in_warp = threadIdx.x % warpSize;

    row_shmem[threadIdx.x] = 0;

    for(int i = thread_idx_in_warp; i < nj; i += warpSize) {
        const int j = irps + i;
        if(j < irpe) {
            row_shmem[threadIdx.x] += as[j] * x[ja[j]];
        }
    }

    if(thread_idx_in_warp == 0) {
        const int warp_local_index = threadIdx.x / warpSize;
        for(int i = 0; i < warpSize; i++) {
            y[warp_global_index] += row_shmem[warp_local_index * warpSize + i];
        }
    }
}