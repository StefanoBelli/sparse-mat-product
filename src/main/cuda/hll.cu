#include <main/cuda/hll.h>

dims_type get_dims_for_hll_v1(int, const cudaDeviceProp&) {

}

__global__ void __kernel_hll_v1(
        const struct ellpack_format *blks, 
        uint64_t numblks, 
        uint32_t hs, 
        uint32_t m, 
        const double *x, 
        double *y) {

    
}