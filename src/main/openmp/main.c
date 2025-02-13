#define _POSIX_C_SOURCE 200809L

//#define LOG_RESULTING_VECTOR

#ifdef LOG_RESULTING_VECTOR
#define LOG_Y_VECTOR(_mtxname, _ch, _m) \
    log_resulting_vector_entries("omp", _mtxname, _ch, _m, y)
#else
#define LOG_Y_VECTOR(_mtxname, _ch, _m)
#endif

#ifndef _OPENMP
#error This code requires OpenMP support (-fopenmp with GCC)
#endif

#include <unistd.h>

#include <omp.h>

#include <utils.h>
#include <executor.h>
#include <matrix/format.h>

static void __kernel_csr(
        const uint64_t *irp, 
        const uint64_t *ja, 
        const double *as, 
        uint32_t m, 
        const double *x, 
        double *y) {
 
#pragma omp parallel for schedule(static)
    for(uint64_t i = 0; i < m; i++) {
        double t = 0;
        for(uint64_t j = irp[i]; j < irp[i + 1]; j++) {
            t += as[j] * x[ja[j]];
        }
        y[i] = t;
    }
}

static void __kernel_hll(
        const struct ellpack_format *blks, 
        uint64_t numblks, 
        uint32_t hs, 
        uint32_t m, 
        const double *x, 
        double *y) {

#pragma omp parallel for schedule(dynamic)
    for(uint64_t numblk = 0; numblk < numblks; numblk++) {
        double t[hs];

        struct ellpack_format ell = blks[numblk];

        for(uint64_t i = 0; i < hs; i++) {
            double ell_tmp = 0;
            for(uint64_t j = 0; j < ell.maxnz; j++) {
                ell_tmp += ell.as[i][j] * x[ell.ja[i][j]];
            }
            t[i] = ell_tmp;
        }

        uint64_t endrow = hs;
        if (numblk * hs + hs > m) {
            endrow = m - numblk * hs;
        }

        for(uint64_t i = 0; i < endrow; i++) {
            y[i + numblk * hs] = t[i];
        }
    }
}

static double 
kernel_hll_caller_taketime(
        const void *format, 
        const union format_args *format_args, 
        const char *mtxname) {

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

    LOG_Y_VECTOR(mtxname, 'h', format_args->hll.m);

    free_reset_ptr(x);
    free_reset_ptr(y); 

    return end - start;
}

static double 
kernel_csr_caller_taketime(
        const void *format, 
        const union format_args *format_args, 
        const char* mtxname) {

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

    LOG_Y_VECTOR(mtxname, 'c', format_args->csr.m);

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
    }

    for(uint32_t i = ncores + 1; i <= 2 * ncores; i++) {
        kexi[i - 1].kernel_time_meter = kernel_hll_caller_taketime;
        kexi[i - 1].format = HLL;
        kexi[i - 1].multiply_datatype = FLOAT64;
        kexi[i - 1].hll_hack_size = 1024;
        kexi[i - 1].cpu_mt_numthreads = i - ncores;
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