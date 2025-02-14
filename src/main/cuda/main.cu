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

    const int load = nj / warpSize;

    row_shmem[threadIdx.x] = 0;

    if(load <= 1) {
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

    uint64_t host_irp[] = { 0, 2, 4, 5, 7 };
    uint64_t host_ja[] = { 0, 1, 1, 2, 2, 2, 3 };
    double host_as[] = { 11, 12, 22, 23, 33, 43, 44 };
    double host_x[] = { 1, 1, 1, 0 };
    double host_y[] = { 0, 0, 0, 0 };

    uint64_t *dev_irp;
    uint64_t *dev_ja;
    double *dev_as;
    double *dev_y;
    double *dev_x;

    checkCudaErrors(cudaMalloc(&dev_irp, sizeof(uint64_t) * 5));
    checkCudaErrors(cudaMalloc(&dev_ja, sizeof(uint64_t) * 7));
    checkCudaErrors(cudaMalloc(&dev_as, sizeof(double) * 7));
    checkCudaErrors(cudaMalloc(&dev_y, sizeof(double) * 4));
    checkCudaErrors(cudaMalloc(&dev_x, sizeof(double) * 4));

    checkCudaErrors(cudaMemcpy(dev_irp, host_irp, sizeof(host_irp), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_ja, host_ja, sizeof(host_ja), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_as, host_as, sizeof(host_as), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_x, host_x, sizeof(host_x), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(dev_y, 0, sizeof(double) * 4));

    checkCudaErrors(cudaFree(dev_irp));
    checkCudaErrors(cudaFree(dev_ja));
    checkCudaErrors(cudaFree(dev_as));
    checkCudaErrors(cudaFree(dev_x));

    checkCudaErrors(cudaMemcpy(host_y, dev_y, sizeof(double) * 4, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dev_y));

    return 0;
}
