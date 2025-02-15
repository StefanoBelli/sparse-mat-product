#include <iostream>
#include <functional>
#include <cmath>

#include <main/cuda/helper_cuda.h>
#include <main/cuda/csr.h>

extern "C" {
#include <matrix/format.h>
#include <executor.h>
#include <utils.h>
}

// need to synchronize
using core_cudakernel_csr_launcher = 
    std::function<void(
        const cudaDeviceProp& devprop,
        const uint64_t* irp,
        const uint64_t* ja,
        const double* as,
        uint32_t m,
        const double* x,
        double* y,
        cudaEvent_t& start, 
        cudaEvent_t& stop)>;

using get_dims_for_csr_cb =
    std::function<dims_type(
        int, 
        const cudaDeviceProp&)>;

using cudakernel_csr_wrapper_cb =
    std::function<void(
        const uint64_t* irp, 
        const uint64_t* ja, 
        const double* as, 
        uint32_t m, 
        const double* x, 
        double* y,
        const dims_type&)>;

static void ensure_device_capabilities_csr(
        const dims_type& dims, 
        const cudaDeviceProp& device_props) {

    unsigned int xgridsz = std::get<0>(dims).x;
    if(xgridsz > (unsigned int) device_props.maxGridSize[0]) {
        std::cerr 
            << "device is unable to handle " 
            << xgridsz
            << " grid dimensionality (x-axis)."
            << " Max allowed is "
            << device_props.maxGridSize[0]
            << std::endl;
        exit(EXIT_FAILURE);
    }

    unsigned int xblocksz = std::get<1>(dims).x;
    if(xblocksz > (unsigned int) device_props.maxThreadsPerBlock) {
        std::cerr 
            << "device is unable to handle " 
            << xblocksz
            << " block dimensionality (x-axis)."
            << " Max allowed is "
            << device_props.maxThreadsPerBlock
            << std::endl;
        exit(EXIT_FAILURE);
    }
    
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

static double base_kernel_csr_caller_taketime(
        const void *format, 
        const union format_args *format_args, 
        const char* mtxname,
        const core_cudakernel_csr_launcher cuda_kernel_launcher) {

    const struct csr_format *csr = (const struct csr_format *) format;

    checkCudaErrors(cudaDeviceReset());

    int device_id;
    checkCudaErrors(cudaGetDevice(&device_id));

    cudaDeviceProp device_props;
    checkCudaErrors(cudaGetDeviceProperties(&device_props, device_id));

    double *host_x = make_vector_of_doubles(format_args->csr.n);

    uint64_t *dev_irp;
    uint64_t *dev_ja;
    double *dev_as;
    double *dev_y;
    double *dev_x;

    size_t irp_size = sizeof(uint64_t) * (format_args->csr.m + 1);
    size_t ja_size = sizeof(uint64_t) * (format_args->csr.nz);
    size_t as_size = sizeof(double) * (format_args->csr.nz);
    size_t y_size = sizeof(double) * (format_args->csr.m);
    size_t x_size = sizeof(double) * (format_args->csr.n);

    checkCudaErrors(cudaMalloc(&dev_irp, irp_size));
    checkCudaErrors(cudaMalloc(&dev_ja, ja_size));
    checkCudaErrors(cudaMalloc(&dev_as, as_size));
    checkCudaErrors(cudaMalloc(&dev_y, y_size));
    checkCudaErrors(cudaMalloc(&dev_x, x_size));

    checkCudaErrors(cudaMemcpy(dev_irp, csr->irp, irp_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_ja, csr->ja, ja_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_as, csr->as, as_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(dev_y, 0, y_size));
    checkCudaErrors(cudaMemcpy(dev_x, host_x, x_size, cudaMemcpyHostToDevice));
    free_reset_ptr(host_x);

    cudaEvent_t start;
    cudaEvent_t stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    cuda_kernel_launcher(
        device_props, dev_irp, dev_ja, dev_as, 
        format_args->csr.m, dev_x, dev_y, start, stop);
    
    float time_ms;
    checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaFree(dev_irp));
    checkCudaErrors(cudaFree(dev_ja));
    checkCudaErrors(cudaFree(dev_as));
    checkCudaErrors(cudaFree(dev_x));

    double *host_y = checked_calloc(double, format_args->csr.m);
    checkCudaErrors(cudaMemcpy(host_y, dev_y, y_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(dev_y));

    // use host_y

    free_reset_ptr(host_y);

    return time_ms / 1000;
}

static double base_kernel_csr_caller_taketime_with_launcher(
        const void *format, 
        const union format_args *format_args, 
        const char* mtxname,
        const get_dims_for_csr_cb& get_dims_for_csr,
        const cudakernel_csr_wrapper_cb& cudakernel) {

    return base_kernel_csr_caller_taketime(
        format, 
        format_args, 
        mtxname, 
        [get_dims_for_csr, cudakernel](
            const cudaDeviceProp& devpr,
            const uint64_t* irp, 
            const uint64_t* ja, 
            const double* as, 
            uint64_t m, 
            const double* x, double* y, 
            const cudaEvent_t& evstart, 
            const cudaEvent_t& evstop){

                dims_type dims = get_dims_for_csr(m, devpr);
                ensure_device_capabilities_csr(dims, devpr);
                checkCudaErrors(cudaEventRecord(evstart, 0));
                cudakernel(irp, ja, as, m, x, y, dims);
                checkCudaErrors(cudaEventRecord(evstop, 0));
                checkCudaErrors(cudaEventSynchronize(evstop));
        });
}

int main() {
    /*
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
    host_x[14] = 2;

    */

    return 0;
}
