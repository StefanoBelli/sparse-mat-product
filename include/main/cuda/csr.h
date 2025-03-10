#ifndef CSR_CUDA_H
#define CSR_CUDA_H

#include <cstdint>

#include <main/cuda/dims_type.h>

dims_type get_dims_for_csr_v1(int, const cudaDeviceProp&);
__global__ void __kernel_csr_v1(
        const uint64_t *, 
        const uint64_t *, 
        const double *, 
        uint32_t, 
        const double *, 
        double *);

dims_type get_dims_for_csr_v2(int, const cudaDeviceProp&);
__global__ void __kernel_csr_v2(
        const uint64_t *, 
        const uint64_t *, 
        const double *, 
        uint32_t, 
        const double *, 
        double *);

dims_type get_dims_for_csr_v3(int, const cudaDeviceProp&);
__global__ void __kernel_csr_v3(
        const uint64_t *, 
        const uint64_t *, 
        const double *, 
        uint32_t, 
        const double *, 
        double *);

#endif