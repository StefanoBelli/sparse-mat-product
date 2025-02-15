#define _POSIX_C_SOURCE 200809L

#include <utils.h>
#include <executor.h>
#include <matrix/format.h>

#include <main/serial/csr.h>
#include <main/serial/hll.h>

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
    
    write_y_vector_to_csv("serial", mtxname, "hll", format_args->hll.m, y);

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

    write_y_vector_to_csv("serial", mtxname, "csr", format_args->csr.m, y);

    free_reset_ptr(x);
    free_reset_ptr(y);

    return end - start;
}

int main(int argc, char** argv) {
    struct kernel_execution_info kexi[2] = {
        {
            .kernel_time_meter = kernel_csr_caller_taketime,
            .format = CSR,
            .multiply_datatype = FLOAT64
        },
        {
            .kernel_time_meter = kernel_hll_caller_taketime,
            .format = HLL,
            .multiply_datatype = FLOAT64,
            .hll_hack_size = 1024,
        }
    };

    struct executor_args eargs = {
        .nkexs = 2,
        .runner = CPU_SERIAL,
        .kexinfos = kexi
    };

    run_executor(argc, argv, &eargs);
    return EXIT_SUCCESS;
}