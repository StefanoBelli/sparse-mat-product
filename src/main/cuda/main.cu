#include <iostream>
#include <algorithm>
#include <cmath>
#include <cuda/helper_cuda.h>

extern "C" {
#include <matrix/format.h>
}

__global__ void __kernel_csr(const uint64_t *irp, const uint64_t *ja, const double *as, uint32_t m, const double *x, double *y) {
    if(threadIdx.x % 32 != 0) {
        return;
    }

    const int warp_global_index = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    
    if(warp_global_index >= m) {
        return;
    }

    double t = 0;

    for(int j = irp[warp_global_index]; j < irp[warp_global_index + 1]; j++) {
        t += as[j] * x[ja[j]];
    }

    y[warp_global_index] = t;
}

__global__ void __kernel_csr_v2(const uint64_t *irp, const uint64_t *ja, const double *as, uint32_t m, const double *x, double *y) {
    extern __shared__ double row_shmem[];

    const int warp_global_index = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if(warp_global_index >= m) {
        return;
    }

    const int irps = irp[warp_global_index];
    const int irpe = irp[warp_global_index + 1];
    const int nj = irpe - irps;

    const int thread_idx_in_warp = threadIdx.x % 32;

    const int load = nj / 32;

    row_shmem[threadIdx.x] = 0;

    if(load <= 1) {
        if(thread_idx_in_warp > nj + 1) {
            return;
        }

        const int j = warp_global_index + thread_idx_in_warp;

        if(irps <= j && j < irpe) {
            row_shmem[threadIdx.x] = as[j] * x[ja[j]];
            printf("thread %d is accessing %d\n", threadIdx.x, j);
        }
    } else {
        for(int i = thread_idx_in_warp; i < nj; i += 32) {
            const int j = warp_global_index + i;
            if(irps <= j && j <= irpe) {
                row_shmem[threadIdx.x] = as[j] * x[ja[j]];
            }
            printf("thread %d del warp %d gestisce indice %d\n", threadIdx.x % 32, threadIdx.x / 32, i);
        }
    }

    __syncthreads();

    if(thread_idx_in_warp == 0) {
        const int warp_local_index = threadIdx.x / 32;
        for(int i = 0; i < 32; i++) {
            y[warp_global_index] += row_shmem[warp_local_index * 32 + i];
        }
    }
}

int numOfWarps(int rows) {
    return rows;
}

int numOfBlocks(int warps) {
    return std::max(1, static_cast<int>(std::ceil(warps / 32)));
}

int main() {
    int device_id;
    checkCudaErrors(cudaGetDevice(&device_id));

    cudaDeviceProp device_props;
    checkCudaErrors(cudaGetDeviceProperties(&device_props, device_id));

    std::cout << device_props.maxGridSize << std::endl;
    std::cout << device_props.sharedMemPerBlock << std::endl;
    std::cout << device_props.maxThreadsPerBlock << std::endl;
    std::cout << device_props.totalGlobalMem  << std::endl;
    std::cout << device_props.maxGridSize[0] << std::endl;
    std::cout << device_props.warpSize << std::endl;

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

    std::cout << "num of blks: " << numOfBlocks(numOfWarps(4)) << "\nnum of warps: " << numOfWarps(4) << '\n';

    dim3 gridSize(numOfBlocks(numOfWarps(4)),1);
    dim3 blockSize(numOfWarps(4) * 32, 1);
    __kernel_csr_v2<<<gridSize, blockSize>>>(dev_irp, dev_ja, dev_as, 4, dev_x, dev_y);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(dev_irp));
    checkCudaErrors(cudaFree(dev_ja));
    checkCudaErrors(cudaFree(dev_as));
    checkCudaErrors(cudaFree(dev_x));

    checkCudaErrors(cudaMemcpy(host_y, dev_y, sizeof(double) * 4, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dev_y));

    for(int i = 0; i < 4; i++) {
        printf(" %lg ", host_y[i]);
    }

    puts("");

    return 0;
}
