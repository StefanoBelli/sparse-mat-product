#include <iostream>
#include <utility>
#include <cmath>
#include <cuda/helper_cuda.h>

extern "C" {
#include <matrix/format.h>
}

void ensure_device_capabilities_csr(const std::pair<dim3, dim3>& dims, const cudaDeviceProp& device_props) {
    if(dims.first.x > device_props.maxGridSize[0]) {
        std::cerr 
            << "device is unable to handle " 
            << dims.first.x 
            << " grid dimensionality (x-axis)."
            << " Max allowed is "
            << device_props.maxGridSize[0]
            << std::endl;
        exit(EXIT_FAILURE);
    }

    if(dims.second.x > device_props.maxThreadsPerBlock) {
        std::cerr 
            << "device is unable to handle " 
            << dims.second.x 
            << " block dimensionality (x-axis)."
            << " Max allowed is "
            << device_props.maxThreadsPerBlock
            << std::endl;
        exit(EXIT_FAILURE);
    }
}

void ensure_device_capabilities_csr(const std::tuple<dim3, dim3, size_t>& dims, const cudaDeviceProp& device_props) {
    ensure_device_capabilities_csr(std::make_pair<>(std::get<0>(dims), std::get<1>(dims)), device_props);

    size_t shmem_size = std::get<2>(dims);
    if(shmem_size > device_props.sharedMemPerBlock) {
        std::cerr
            << "device is unable to handle "
            << shmem_size
            << "B of shared memory amount per block."
            << " Max allowed is "
            << device_props.sharedMemPerBlock
            << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::pair<dim3, dim3> get_dims_for_csr_v1(int nrows, const cudaDeviceProp& device_props) {
    int max_thr_per_blk = device_props.maxThreadsPerBlock;

    if(nrows <= max_thr_per_blk) {
        dim3 grid_dim(1, 1);
        dim3 block_dim(nrows, 1);
        return std::make_pair<>(grid_dim, block_dim);
    }

    double splsz = nrows / max_thr_per_blk;
    int xgridsz = nrows % max_thr_per_blk ? ceil(splsz) + 1 : splsz;

    dim3 grid_dim(xgridsz, 1);
    dim3 block_dim(max_thr_per_blk, 1);
    return std::make_pair<>(grid_dim, block_dim);
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

std::pair<dim3, dim3> get_dims_for_csr_v2(int nrows, const cudaDeviceProp& device_props) {
    int warp_size = device_props.warpSize;
    int max_thr_per_blk = device_props.maxThreadsPerBlock;

    if(nrows * warp_size <= max_thr_per_blk) {
        dim3 grid_dim(1,1);
        dim3 block_dim(nrows * warp_size, 1);
        return std::make_pair<>(grid_dim, block_dim);
    }

    double splsz = nrows * warp_size / max_thr_per_blk;
    int xgridsz = (nrows * warp_size) % max_thr_per_blk ? ceil(splsz) + 1 : splsz;

    dim3 grid_dim(xgridsz, 1);
    dim3 block_dim(max_thr_per_blk, 1);
    return std::make_pair<>(grid_dim, block_dim);
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

std::tuple<dim3, dim3, size_t> get_dims_for_csr_v3(int nrows, const cudaDeviceProp& device_props) {
    auto csrv2dims = get_dims_for_csr_v2(nrows, device_props);
    size_t shmemsize = csrv2dims.second.x * sizeof(double); 
    return std::make_tuple<>(csrv2dims.first, csrv2dims.second, shmemsize);
}

__global__ void __kernel_csr_v3(
        const uint64_t *irp, 
        const uint64_t *ja, 
        const double *as, 
        int32_t m, 
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

    if(nj <= warpSize) {
        if(thread_idx_in_warp > nj + 1) {
            return;
        }

        const int j = warp_global_index + thread_idx_in_warp;

        if(irps <= j && j < irpe) {
            row_shmem[threadIdx.x] = as[j] * x[ja[j]];
        }
    } else {
        for(int i = thread_idx_in_warp; i < nj; i += warpSize) {
            const int j = warp_global_index + i;
            if(irps <= j && j < irpe) {
                row_shmem[threadIdx.x] += as[j] * x[ja[j]];
            }
        }
    }

    __syncthreads();

    if(thread_idx_in_warp == 0) {
        const int warp_local_index = threadIdx.x / warpSize;
        for(int i = 0; i < warpSize; i++) {
            y[warp_global_index] += row_shmem[warp_local_index * warpSize + i];
        }
    }
}

int main() {

    int device_id;
    checkCudaErrors(cudaGetDevice(&device_id));

    cudaDeviceProp device_props;
    checkCudaErrors(cudaGetDeviceProperties(&device_props, device_id));

    cudaDeviceReset();

    int m = 48;
    int n = 48;

    struct coo_format coo[] = {
        { .i = 0, .j = 0, .v = 1 },
        { .i = 1, .j = 1, .v = 2 },
        { .i = 2, .j = 2, .v = 3 },
        { .i = 3, .j = 3, .v = 4 },
        { .i = 4, .j = 4, .v = 5 },
        { .i = 5, .j = 5, .v = 6 },
        { .i = 6, .j = 6, .v = 1 },
        { .i = 7, .j = 7, .v = 2 },
        { .i = 8, .j = 8, .v = 3 },
        { .i = 9, .j = 9, .v = 4 },
        { .i = 10, .j = 10, .v = 5 },
        { .i = 11, .j = 11, .v = 6 },
        { .i = 12, .j = 12, .v = 1 },
        { .i = 13, .j = 13, .v = 2 },
        { .i = 14, .j = 14, .v = 3 },
        { .i = 15, .j = 15, .v = 4 },
        { .i = 16, .j = 16, .v = 5 },
        { .i = 17, .j = 17, .v = 6 },
        { .i = 18, .j = 18, .v = 1 },
        { .i = 19, .j = 19, .v = 2 },
        { .i = 20, .j = 20, .v = 3 },
        { .i = 21, .j = 21, .v = 4 },
        { .i = 22, .j = 22, .v = 5 },
        { .i = 23, .j = 23, .v = 6 },
        { .i = 24, .j = 24, .v = 1 },
        { .i = 25, .j = 25, .v = 2 },
        { .i = 26, .j = 26, .v = 3 },
        { .i = 27, .j = 27, .v = 4 },
        { .i = 28, .j = 28, .v = 5 },
        { .i = 29, .j = 29, .v = 6 },
        { .i = 30, .j = 30, .v = 1 },
        { .i = 31, .j = 31, .v = 2 },
        { .i = 32, .j = 32, .v = 3 },
        { .i = 33, .j = 33, .v = 4 },
        { .i = 34, .j = 34, .v = 5 },
        { .i = 35, .j = 35, .v = 6 },
        { .i = 36, .j = 36, .v = 1 },
        { .i = 37, .j = 37, .v = 2 },
        { .i = 38, .j = 38, .v = 3 },
        { .i = 39, .j = 39, .v = 4 },
        { .i = 40, .j = 40, .v = 5 },
        { .i = 41, .j = 41, .v = 6 },
        { .i = 42, .j = 42, .v = 1 },
        { .i = 43, .j = 43, .v = 2 },
        { .i = 44, .j = 44, .v = 3 },
        { .i = 45, .j = 45, .v = 4 },
        { .i = 46, .j = 46, .v = 5 },

        { .i = 47, .j = 47, .v = 1 },
        { .i = 47, .j = 46, .v = 1 },
        { .i = 47, .j = 45, .v = 1 },
        { .i = 47, .j = 44, .v = 1 },
        { .i = 47, .j = 43, .v = 1 },
        { .i = 47, .j = 42, .v = 1 },
        { .i = 47, .j = 41, .v = 1 },
        { .i = 47, .j = 40, .v = 1 },
        { .i = 47, .j = 39, .v = 1 },
        { .i = 47, .j = 38, .v = 1 },
        { .i = 47, .j = 37, .v = 1 },
        { .i = 47, .j = 36, .v = 1 },
        { .i = 47, .j = 35, .v = 1 },
        { .i = 47, .j = 34, .v = 1 },
        { .i = 47, .j = 33, .v = 1 },
        { .i = 47, .j = 32, .v = 1 },
        { .i = 47, .j = 31, .v = 1 },
        { .i = 47, .j = 30, .v = 1 },
        { .i = 47, .j = 29, .v = 1 },
        { .i = 47, .j = 28, .v = 1 },
        { .i = 47, .j = 27, .v = 1 },
        { .i = 47, .j = 26, .v = 1 },
        { .i = 47, .j = 25, .v = 1 },
        { .i = 47, .j = 24, .v = 1 },
        { .i = 47, .j = 23, .v = 1 },
        { .i = 47, .j = 22, .v = 1 },
        { .i = 47, .j = 21, .v = 1 },
        { .i = 47, .j = 20, .v = 1 },
        { .i = 47, .j = 19, .v = 1 },
        { .i = 47, .j = 18, .v = 1 },
        { .i = 47, .j = 17, .v = 1 },
        { .i = 47, .j = 16, .v = 1 },
        { .i = 47, .j = 15, .v = 1 },
        { .i = 47, .j = 14, .v = 1 },
    };

    double host_x[48];

    memset(host_x, 0, sizeof(host_x));

    host_x[47] = 1;
    host_x[46] = 1;
    host_x[45] = 1;
    host_x[44] = 1;
    host_x[43] = 1;
    host_x[42] = 1;
    host_x[41] = 1;
    host_x[40] = 1;
    host_x[39] = 1;
    host_x[38] = 1;
    host_x[37] = 1;
    host_x[36] = 1;
    host_x[35] = 1;
    host_x[34] = 1;
    host_x[33] = 1;
    host_x[32] = 1;
    host_x[31] = 1;
    host_x[30] = 1;
    host_x[29] = 1;
    host_x[28] = 1;
    host_x[27] = 1;
    host_x[26] = 1;
    host_x[25] = 1;
    host_x[24] = 1;
    host_x[23] = 1;
    host_x[22] = 1;
    host_x[21] = 1;
    host_x[20] = 1;
    host_x[19] = 1;
    host_x[18] = 1;
    host_x[17] = 1;
    host_x[16] = 1;
    host_x[15] = 1;
    host_x[14] = 1;

    /* ---- */

    int nz = sizeof(coo) / sizeof(struct coo_format);
    struct csr_format csr;
    coo_to_csr(&csr, coo, nz, m);

    uint64_t *dev_irp;
    uint64_t *dev_ja;
    double *dev_as;
    double *dev_y;
    double *dev_x;

    checkCudaErrors(cudaMalloc(&dev_irp, sizeof(uint64_t) * (m + 1)));
    checkCudaErrors(cudaMalloc(&dev_ja, sizeof(uint64_t) * (nz)));
    checkCudaErrors(cudaMalloc(&dev_as, sizeof(double) * (nz)));
    checkCudaErrors(cudaMalloc(&dev_y, sizeof(double) * (m)));
    checkCudaErrors(cudaMalloc(&dev_x, sizeof(host_x)));

    checkCudaErrors(cudaMemcpy(dev_irp, csr.irp, sizeof(uint64_t) * (m + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_ja, csr.ja, sizeof(uint64_t) * (nz), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_as, csr.as, sizeof(double) * (nz), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_x, host_x, sizeof(host_x), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(dev_y, 0, sizeof(double) * (m)));

    auto csr_v3_dims = get_dims_for_csr_v3(m, device_props);
    ensure_device_capabilities_csr(csr_v3_dims, device_props);

    cudaEvent_t start;
    cudaEvent_t stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));
    __kernel_csr_v3<<<std::get<0>(csr_v3_dims), std::get<1>(csr_v3_dims), std::get<2>(csr_v3_dims)>>>(dev_irp, dev_ja, dev_as, m, dev_x, dev_y);
    checkCudaErrors(cudaEventRecord(stop, 0));

    checkCudaErrors(cudaDeviceSynchronize());

    float timeMs;
    checkCudaErrors(cudaEventElapsedTime(&timeMs, start, stop));
    std::cout << timeMs / 1000 << " s" << std::endl;

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    /*
    auto csr_v1_dims = get_dims_for_csr_v1(m, device_props);
    ensure_device_capabilities_csr(csr_v1_dims, device_props);
    __kernel_csr_v1<<<csr_v1_dims.first,csr_v1_dims.second>>>(dev_irp, dev_ja, dev_as, m, dev_x, dev_y);
    checkCudaErrors(cudaDeviceSynchronize());

    auto csr_v2_dims = get_dims_for_csr_v2(m, device_props);
    ensure_device_capabilities_csr(csr_v2_dims, device_props);
    __kernel_csr_v2<<<csr_v2_dims.first,csr_v2_dims.second>>>(dev_irp, dev_ja, dev_as, m, dev_x, dev_y);
    checkCudaErrors(cudaDeviceSynchronize());
    */


    checkCudaErrors(cudaFree(dev_irp));
    checkCudaErrors(cudaFree(dev_ja));
    checkCudaErrors(cudaFree(dev_as));
    checkCudaErrors(cudaFree(dev_x));

    double host_y[m];

    checkCudaErrors(cudaMemcpy(host_y, dev_y, sizeof(double) * (m), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dev_y));

    puts("");
    for(int i = 0; i < m; i++) {
        printf("y[%d] = %lg\n", i, host_y[i]);
    }
    puts("");

    return 0;
}
