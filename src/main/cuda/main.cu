#include <iostream>
#include <functional>

#include <main/cuda/helper_cuda.h>
#include <main/cuda/csr.h>

extern "C" {
#include <matrix/format.h>
#include <executor.h>
#include <utils.h>
}

/**
 * CSR launchers
 */
using cudakernel_csr_launcher_cb = 
    std::function<void(
        const cudaDeviceProp& devprop,
        const uint64_t* irp,
        const uint64_t* ja,
        const double* as,
        uint32_t m,
        const double* x,
        double* y,
        const cudaEvent_t& start, 
        const cudaEvent_t& stop)>;

using get_dims_for_csr_cb =
    std::function<dims_type(
        int nrows, 
        const cudaDeviceProp& devprop)>;

using cudakernel_csr_call_wrapper_cb =
    std::function<void(
        const uint64_t* irp, 
        const uint64_t* ja, 
        const double* as, 
        uint32_t m, 
        const double* x, 
        double* y,
        const cudaEvent_t& start,
        const cudaEvent_t& stop,
        const dims_type& dims)>;

static void 
ensure_device_capabilities_csr(
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

static double 
base_kernel_csr_caller_taketime(
        const void *format, 
        const union format_args *format_args, 
        const char* mtxname,
        const cudakernel_csr_launcher_cb& cuda_kernel_launcher,
        const char* variant,
        mult_datatype multiply_datatype) {

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

    write_y_vector_to_csv("gpu", variant, multiply_datatype, mtxname, "csr", format_args->csr.m, host_y);

    free_reset_ptr(host_y);

    return time_ms / 1000;
}

#define CUDAKERNEL_LAUNCHER_CSR_ARGLIST \
    const uint64_t* irp, \
    const uint64_t* ja, \
    const double* as, \
    uint64_t m, \
    const double* x, \
    double* y, \
    const cudaEvent_t& evstart, \
    const cudaEvent_t& evstop

#define CUDAKERNEL_WRAPPER_CSR_ARGLIST \
    CUDAKERNEL_LAUNCHER_CSR_ARGLIST, \
    const dims_type& dims

static double 
base_kernel_csr_caller_taketime_with_launcher(
        const void *format, 
        const union format_args *format_args, 
        const char* mtxname,
        const char* variant,
        const mult_datatype multiply_datatype,
        const get_dims_for_csr_cb& get_dims_for_csr,
        const cudakernel_csr_call_wrapper_cb& cudakernel) {

    return base_kernel_csr_caller_taketime(
        format, 
        format_args, 
        mtxname, 
        [get_dims_for_csr, cudakernel](
            const cudaDeviceProp& devpr,
            CUDAKERNEL_LAUNCHER_CSR_ARGLIST){

                dims_type dims = get_dims_for_csr(m, devpr);
                ensure_device_capabilities_csr(dims, devpr);
                cudakernel(irp, ja, as, m, x, y, 
                            evstart, evstop, dims);
                checkCudaErrors(cudaEventSynchronize(evstop));
        },
        variant,
        multiply_datatype);
}

static double 
kernel_csr_v1_caller_taketime(
        const void* format, 
        const union format_args *format_args, 
        const char* mtxname,
        mult_datatype multiply_datatype,
        const char* variant) {

    return base_kernel_csr_caller_taketime_with_launcher(
        format, format_args, mtxname, variant, multiply_datatype,
        get_dims_for_csr_v1,
        [](CUDAKERNEL_WRAPPER_CSR_ARGLIST) {
            checkCudaErrors(cudaEventRecord(evstart));
            __kernel_csr_v1<<<
                std::get<0>(dims), 
                std::get<1>(dims), 
                std::get<2>(dims)>>>
                (irp, ja, as, m, x, y);
            checkCudaErrors(cudaEventRecord(evstop));
        }
    );
}

static double 
kernel_csr_v2_caller_taketime(
        const void* format, 
        const union format_args *format_args, 
        const char* mtxname,
        mult_datatype multiply_datatype,
        const char* variant) {

    return base_kernel_csr_caller_taketime_with_launcher(
        format, format_args, mtxname, variant, multiply_datatype,
        get_dims_for_csr_v2,
        [](CUDAKERNEL_WRAPPER_CSR_ARGLIST) {
            checkCudaErrors(cudaEventRecord(evstart));
            __kernel_csr_v2<<<
                std::get<0>(dims), 
                std::get<1>(dims), 
                std::get<2>(dims)>>>
                (irp, ja, as, m, x, y);
            checkCudaErrors(cudaEventRecord(evstop));
        }
    );
}

static double 
kernel_csr_v3_caller_taketime(
        const void* format, 
        const union format_args *format_args, 
        const char* mtxname,
        mult_datatype multiply_datatype,
        const char* variant) {

    return base_kernel_csr_caller_taketime_with_launcher(
        format, format_args, mtxname, variant, multiply_datatype,
        get_dims_for_csr_v3,
        [](CUDAKERNEL_WRAPPER_CSR_ARGLIST) {
            checkCudaErrors(cudaEventRecord(evstart));
            __kernel_csr_v3<<<
                std::get<0>(dims), 
                std::get<1>(dims), 
                std::get<2>(dims)>>>
                (irp, ja, as, m, x, y);
            checkCudaErrors(cudaEventRecord(evstop));
        }
    );
}

#undef CUDAKERNEL_WRAPPER_CSR_ARGLIST
#undef CUDAKERNEL_LAUNCHER_CSR_ARGLIST

/**
 * HLL launcher
 */


using cudakernel_hll_launcher_cb = 
    std::function<void(
        const cudaDeviceProp& devprop,
        const struct ellpack_format* blks,
        uint64_t numblks,
        uint32_t hs,
        uint32_t m,
        const double* x,
        double* y,
        const cudaEvent_t& start, 
        const cudaEvent_t& stop)>;

static void
base_kernel_hll_caller_taketime(
        const void *format, 
        const union format_args *format_args, 
        const char* mtxname,
        const cudakernel_hll_launcher_cb& cuda_kernel_launcher,
        const char* variant,
        mult_datatype multiply_datatype) {

    const struct hll_format *old_hll = (const struct hll_format*) format;

    struct hll_format hll;
    transpose_hll(&hll, old_hll, format_args->hll.hs);

    free_hll_format(&hll, format_args->hll.hs);
}


static double 
test(
        const void* format,
        const union format_args *format_args,
        const char* e,
        mult_datatype m,
        const char* v) {

    const struct hll_format *old_hll = (const struct hll_format*) format;

    struct hll_format hll;
    transpose_hll(&hll, old_hll, format_args->hll.hs);

    free_transposed_hll_format(&hll);

    return 0;
        }

int main(int argc, char** argv) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

    struct kernel_execution_info kexi[1] = {
        {
            .kernel_time_meter = test,
            .format = HLL,
            .multiply_datatype = FLOAT64,
        }
        /*
        {
            .kernel_time_meter = kernel_csr_v1_caller_taketime,
            .format = CSR,
            .multiply_datatype = FLOAT64,
            .variant_name = "v1"
        },
        {
            .kernel_time_meter = kernel_csr_v2_caller_taketime,
            .format = CSR,
            .multiply_datatype = FLOAT64,
            .variant_name = "v2"
        },
        {
            .kernel_time_meter = kernel_csr_v3_caller_taketime,
            .format = CSR,
            .multiply_datatype = FLOAT64,
            .variant_name = "v3"
        },
        */
    };

    struct executor_args eargs = {
        .kexinfos = kexi,
        .nkexs = 1,
        .runner = GPU,
    };

#pragma GCC diagnostic pop

    run_executor(argc, argv, &eargs);
    return EXIT_SUCCESS;
}
