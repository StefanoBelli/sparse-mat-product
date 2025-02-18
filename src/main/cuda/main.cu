#include <iostream>
#include <functional>

#include <main/cuda/helper_cuda.h>
#include <main/cuda/csr.h>
#include <main/cuda/hll.h>

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
        const double **as,
        const uint64_t **ja,
        const uint64_t *maxnzs,
        const size_t *pitches_as,
        const size_t *pitches_ja,
        uint64_t numblks,
        uint32_t hs,
        uint32_t m,
        const double* x,
        double* y,
        const cudaEvent_t& start, 
        const cudaEvent_t& stop)>;

using get_dims_for_hll_cb =
    std::function<dims_type(
        int nrows, 
        const cudaDeviceProp& devprop)>;

using cudakernel_hll_call_wrapper_cb =
    std::function<void(
        const double **as,
        const uint64_t **ja,
        const uint64_t *maxnzs,
        const size_t *pitches_as,
        const size_t *pitches_ja,
        uint64_t numblks,
        uint32_t hs,
        uint32_t m, 
        const double* x, 
        double* y,
        const cudaEvent_t& start,
        const cudaEvent_t& stop,
        const dims_type& dims)>;

static void 
ensure_device_capabilities_hll(
        const dims_type& dims, 
        const cudaDeviceProp& device_props) {

}

static double
base_kernel_hll_caller_taketime(
        const void *format, 
        const union format_args *format_args, 
        const char* mtxname,
        const cudakernel_hll_launcher_cb& cuda_kernel_launcher,
        const char* variant,
        mult_datatype multiply_datatype) {

    const struct hll_format *old_hll = (const struct hll_format*) format;

    checkCudaErrors(cudaDeviceReset());

    int device_id;
    checkCudaErrors(cudaGetDevice(&device_id));

    cudaDeviceProp device_props;
    checkCudaErrors(cudaGetDeviceProperties(&device_props, device_id));

    struct hll_format hll;
    contig_transposed_hll(&hll, old_hll, format_args->hll.hs);

    double *host_x = make_vector_of_doubles(format_args->hll.n);
    size_t *host_pitches_as = checked_malloc(size_t, hll.numblks);
    size_t *host_pitches_ja = checked_malloc(size_t, hll.numblks);
    double **host_dev_as = checked_malloc(double*, hll.numblks);
    uint64_t **host_dev_ja = checked_malloc(uint64_t*, hll.numblks);

    double **dev_as;
    uint64_t **dev_ja;
    uint64_t *dev_maxnzs;
    size_t *dev_pitches_as;
    size_t *dev_pitches_ja;
    double *dev_y;
    double *dev_x;

    size_t x_size = sizeof(double) * format_args->hll.n;
    size_t maxnzs_size = sizeof(uint64_t) * hll.numblks;
    size_t y_size = sizeof(double) * format_args->hll.m;
    size_t pitches_size = sizeof(size_t) * hll.numblks;
    size_t as_size = sizeof(double*) * hll.numblks;
    size_t ja_size = sizeof(uint64_t*) * hll.numblks;

    checkCudaErrors(cudaMalloc(&dev_y, y_size));
    checkCudaErrors(cudaMalloc(&dev_x, x_size));
    checkCudaErrors(cudaMalloc(&dev_pitches_ja, pitches_size));
    checkCudaErrors(cudaMalloc(&dev_pitches_as, pitches_size));
    checkCudaErrors(cudaMalloc(&dev_as, as_size));
    checkCudaErrors(cudaMalloc(&dev_ja, ja_size));
    checkCudaErrors(cudaMalloc(&dev_maxnzs, maxnzs_size));

    for(uint64_t i = 0; i < hll.numblks; i++) {
        checkCudaErrors(cudaMallocPitch(
            &host_dev_as[i], 
            &host_pitches_as[i], 
            format_args->hll.hs * sizeof(double), 
            hll.blks[i].maxnz));
        checkCudaErrors(cudaMallocPitch(
            &host_dev_ja[i], 
            &host_pitches_ja[i], 
            format_args->hll.hs * sizeof(uint64_t), 
            hll.blks[i].maxnz));
    }

    checkCudaErrors(cudaMemcpy(dev_pitches_as, host_pitches_as, pitches_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_pitches_ja, host_pitches_ja, pitches_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(dev_as, host_dev_as, as_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_ja, host_dev_ja, ja_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(dev_y, 0, y_size));
    checkCudaErrors(cudaMemcpy(dev_x, host_x, x_size, cudaMemcpyHostToDevice));

    for(uint64_t i = 0; i < hll.numblks; i++) {
        checkCudaErrors(cudaMemcpy(
            &dev_maxnzs[i], 
            &hll.blks[i].maxnz, 
            sizeof(uint64_t), 
            cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy2D(
            host_dev_as[i], 
            host_pitches_as[i], 
            hll.blks[i].as, 
            format_args->hll.hs * sizeof(double), 
            format_args->hll.hs * sizeof(double), 
            hll.blks[i].maxnz, 
            cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy2D(
            host_dev_ja[i], 
            host_pitches_ja[i], 
            hll.blks[i].ja, 
            format_args->hll.hs * sizeof(uint64_t), 
            format_args->hll.hs * sizeof(uint64_t), 
            hll.blks[i].maxnz, 
            cudaMemcpyHostToDevice));
    }

    free_reset_ptr(host_pitches_as);
    free_reset_ptr(host_pitches_ja);
    
    free_reset_ptr(host_x);

    cudaEvent_t start;
    cudaEvent_t stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    cuda_kernel_launcher(
        device_props, const_cast<const double**>(dev_as), 
        const_cast<const uint64_t**>(dev_ja), dev_maxnzs, 
        dev_pitches_as, dev_pitches_ja, hll.numblks, 
        format_args->hll.hs, format_args->hll.m, 
        dev_x, dev_y, start, stop);

    float time_ms;
    checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    for(uint64_t i = 0; i < hll.numblks; i++) {
        checkCudaErrors(cudaFree(host_dev_as[i]));
    }
    checkCudaErrors(cudaFree(dev_as));
    free_reset_ptr(host_dev_as);

    for(uint64_t i = 0; i < hll.numblks; i++) {
        checkCudaErrors(cudaFree(host_dev_ja[i]));
    }
    checkCudaErrors(cudaFree(dev_ja));
    free_reset_ptr(host_dev_ja);

    free_contig_transposed_hll_format(&hll);

    checkCudaErrors(cudaFree(dev_maxnzs));
    checkCudaErrors(cudaFree(dev_pitches_as));
    checkCudaErrors(cudaFree(dev_pitches_ja));
    checkCudaErrors(cudaFree(dev_x));
    double *host_y = checked_calloc(double, format_args->hll.m);
    checkCudaErrors(cudaMemcpy(host_y, dev_y, y_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(dev_y));

    write_y_vector_to_csv("gpu", variant, multiply_datatype, mtxname, "hll", format_args->hll.m, host_y);

    free_reset_ptr(host_y);

    return time_ms / 1000;
}

#define CUDAKERNEL_LAUNCHER_HLL_ARGLIST \
    const double **as, \
    const uint64_t **ja, \
    const uint64_t *maxnzs, \
    const size_t *pitches_as, \
    const size_t *pitches_ja, \
    uint64_t numblks, \
    uint32_t hs, \
    uint32_t m, \
    const double *x, \
    double *y, \
    const cudaEvent_t& evstart, \
    const cudaEvent_t& evstop

#define CUDAKERNEL_WRAPPER_HLL_ARGLIST \
    CUDAKERNEL_LAUNCHER_HLL_ARGLIST, \
    const dims_type& dims

static double 
base_kernel_hll_caller_taketime_with_launcher(
        const void *format, 
        const union format_args *format_args, 
        const char* mtxname,
        const char* variant,
        const mult_datatype multiply_datatype,
        const get_dims_for_hll_cb& get_dims_for_hll,
        const cudakernel_hll_call_wrapper_cb& cudakernel) {

    return base_kernel_hll_caller_taketime(
        format, 
        format_args, 
        mtxname, 
        [get_dims_for_hll, cudakernel](
            const cudaDeviceProp& devpr,
            CUDAKERNEL_LAUNCHER_HLL_ARGLIST){
                
                dims_type dims = get_dims_for_hll(numblks, devpr);
                ensure_device_capabilities_hll(dims, devpr);
                cudakernel(as, ja, maxnzs, pitches_as, pitches_ja, 
                            numblks, hs, m, x, y, evstart, evstop, dims);
                checkCudaErrors(cudaEventSynchronize(evstop));
        },
        variant,
        multiply_datatype);
}

static double 
kernel_hll_v1_caller_taketime(
        const void* format, 
        const union format_args *format_args, 
        const char* mtxname,
        mult_datatype multiply_datatype,
        const char* variant) {

    return base_kernel_hll_caller_taketime_with_launcher(
        format, format_args, mtxname, variant, multiply_datatype,
        get_dims_for_hll_v1,
        [](CUDAKERNEL_WRAPPER_HLL_ARGLIST) {
            checkCudaErrors(cudaEventRecord(evstart));
            __kernel_hll_v1<<<
                std::get<0>(dims), 
                std::get<1>(dims), 
                std::get<2>(dims)>>>
                (as, ja, maxnzs, pitches_as,
                pitches_ja, numblks, hs, m, x, y);
            checkCudaErrors(cudaEventRecord(evstop));
        }
    );
}

#undef CUDAKERNEL_LAUNCHER_HLL_ARGLIST
#undef CUDAKERNEL_WRAPPER_HLL_ARGLIST

int main(int argc, char** argv) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

    struct kernel_execution_info kexi[4] = {
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
        {
            .kernel_time_meter = kernel_hll_v1_caller_taketime,
            .format = HLL,
            .hll_hack_size = 32,
            .multiply_datatype = FLOAT64,
        }
    };

    struct executor_args eargs = {
        .kexinfos = kexi,
        .nkexs = 4,
        .runner = GPU,
    };

#pragma GCC diagnostic pop

    run_executor(argc, argv, &eargs);
    return EXIT_SUCCESS;
}
