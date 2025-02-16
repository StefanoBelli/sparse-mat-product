#define _POSIX_C_SOURCE 200809L

#ifndef _OPENMP
#error This code requires OpenMP support (-fopenmp with GCC)
#endif

#include <omp.h>

#include <unistd.h>

#include <utils.h>
#include <executor.h>
#include <matrix/format.h>

#include <main/openmp/csr.h>
#include <main/openmp/hll.h>

static double 
kernel_hll_caller_taketime(
        const void *format, 
        const union format_args *format_args, 
        const char *mtxname,
        mult_datatype multiply_datatype,
        const char* variant) {

    const struct hll_format *hll = (const struct hll_format*) format;

    uint32_t hs = format_args->hll.hs;
    uint32_t m = format_args->hll.m;

    struct ellpack_format *blks = hll->blks;
    uint64_t numblks = hll->numblks;

    double *x = make_vector_of_doubles(format_args->csr.n);
    double *y = checked_calloc(double, m);

    double start = hrt_get_time();
    __kernel_hll(blks, numblks, hs, m, x, y);
    double end = hrt_get_time();

    if(omp_get_max_threads() == 2) { 
        write_y_vector_to_csv("omp", variant, multiply_datatype, mtxname, "hll", format_args->hll.m, y);
    }

    free_reset_ptr(x);
    free_reset_ptr(y); 

    return end - start;
}

static double 
kernel_csr_caller_taketime(
        const void *format, 
        const union format_args *format_args, 
        const char* mtxname,
        mult_datatype multiply_datatype,
        const char* variant) {

    const struct csr_format *csr = (const struct csr_format*) format;

    uint32_t m = format_args->csr.m;
    double *as = csr->as;
    uint64_t *irp = csr->irp;
    uint64_t *ja = csr->ja;

    double *x = make_vector_of_doubles(format_args->csr.n);
    double *y = checked_calloc(double, format_args->csr.m);

    double start = hrt_get_time();
    __kernel_csr(irp, ja, as, m, x, y);
    double end = hrt_get_time();

    if(omp_get_max_threads() == 2) { 
        write_y_vector_to_csv("omp", variant, multiply_datatype, mtxname, "csr", format_args->csr.m, y);
    }

    free_reset_ptr(x);
    free_reset_ptr(y);

    return end - start;
}

static int
set_num_threads_omp(uint32_t thrs) {
    omp_set_num_threads(thrs);
    return !(omp_get_max_threads() == (int) thrs);
}

int main(int argc, char** argv) {
    long ncores = sysconf(_SC_NPROCESSORS_ONLN);
    if(ncores < 1) {
        log_error(sysconf);
        return EXIT_FAILURE;
    }

    printf("num of cores online = %ld\n", ncores);

    struct kernel_execution_info *kexi = checked_calloc(struct kernel_execution_info, 2 * ncores);

    for(uint32_t i = 1; i <= ncores; i++) {
        kexi[i - 1].kernel_time_meter = kernel_csr_caller_taketime;
        kexi[i - 1].format = CSR;
        kexi[i - 1].multiply_datatype = FLOAT64;
        kexi[i - 1].cpu_mt_numthreads = i;
        kexi[i - 1].variant_name = NULL;
    }

    for(uint32_t i = ncores + 1; i <= 2 * ncores; i++) {
        kexi[i - 1].kernel_time_meter = kernel_hll_caller_taketime;
        kexi[i - 1].format = HLL;
        kexi[i - 1].multiply_datatype = FLOAT64;
        kexi[i - 1].hll_hack_size = 1024;
        kexi[i - 1].cpu_mt_numthreads = i - ncores;
        kexi[i - 1].variant_name = NULL;
    }

    struct executor_args eargs = {
        .nkexs = 2 * ncores,
        .runner = CPU_MT,
        .kexinfos = kexi,
        .set_num_thread = set_num_threads_omp
    };

    run_executor(argc, argv, &eargs);

    free_reset_ptr(kexi);

    return EXIT_SUCCESS;
}