#define _POSIX_C_SOURCE 200809L

#include <utils.h>
#include <executor.h>
#include <matrix/format.h>
#include <linux/limits.h>

#define always_inline inline __attribute__((always_inline))

static double kernel_csr(const void *format, const union format_args *format_args, const char* mtxname) {
    const struct csr_format *csr = (const struct csr_format*) format;

    double *x = make_vector_of_doubles(format_args->csr.n);
    double *y = checked_calloc(double, format_args->csr.m);

    double start = hrt_get_time();
    for(uint64_t i = 0; i < format_args->csr.m; i++) {
        for(uint64_t j = csr->irp[i]; j < csr->irp[i + 1]; j++) {
            y[i] += csr->as[j] * x[csr->ja[j]];
        }
    }
    double end = hrt_get_time();

    free_reset_ptr(x);
    //free_reset_ptr(y);

    char buf[PATH_MAX];
    snprintf(buf, PATH_MAX, "%s_c.log", mtxname);
    FILE *f = fopen(buf, "w");
    for(int i = 0; i < format_args->csr.m; i++) {
        fprintf(f, "%lg\n", y[i]);
    }

    free_reset_ptr(y);
    fclose(f);

    return end - start;
}

static always_inline void ellprod(double* y, uint64_t m, const struct ellpack_format* ell, const double* x) {
    for(uint64_t i = 0; i < m; i++) {
        for(uint64_t j = 0; j < ell->maxnz; j++) {
            y[i] += ell->as[i][j] * x[ell->ja[i][j]];
        }
    }
}

static double kernel_hll(const void *format, const union format_args *format_args, const char *mtxname) {
    const struct hll_format *hll = (const struct hll_format*) format;

    uint32_t hs = format_args->hll.hs;
    uint32_t m = format_args->hll.m;

    double *x = make_vector_of_doubles(format_args->csr.n);
    double *y = checked_calloc(double, m);

    double *t = checked_calloc(double, hs);

    double start = hrt_get_time();
    for(uint64_t numblk = 0; numblk < hll->numblks; numblk++) {
        ellprod(t, hs, &hll->blks[numblk], x);
        uint64_t endrow = hs;
        if (numblk * hs + hs > m) {
            endrow = m - numblk * hs;
        }

        for(uint64_t i = 0; i < endrow; i++) {
            y[i + numblk * hs] = t[i];
            t[i] = 0;
        }
    }
    double end = hrt_get_time();

    free_reset_ptr(t);
    free_reset_ptr(x);
    //free_reset_ptr(y);

    char buf[PATH_MAX];
    snprintf(buf, PATH_MAX, "%s_h.log", mtxname);
    FILE *f = fopen(buf, "w");
    for(int i = 0; i < format_args->csr.m; i++) {
        fprintf(f, "%lg\n", y[i]);
    }

    free_reset_ptr(y);
    fclose(f);
    return end - start;
}

int main(int argc, char** argv) {
    struct kernel_execution_info kexi[2] = {
        {
            .kernel = kernel_csr,
            .format = CSR
        },
        {
            .kernel = kernel_hll,
            .format = HLL,
            .hll_hack_size = 32
        }
    };

    struct executor_args eargs = {
        .nkexs = 2,
        .runner = CPU_SERIAL,
        .kexinfos = kexi
    };

    run_executor(argc, argv, &eargs);
    return 0;
}