#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <stdint.h>

enum mult_datatype {
    FLOAT64,
    FLOAT32,
    INT32,
    INT64
};

enum matrix_format {
    CSR,
    HLL
};

enum runner_type {
    CPU_SERIAL,
    CPU_MT,
    GPU
};

struct csr_args {
    uint64_t m;
    uint64_t nz;
};

struct hll_args {
    uint64_t m;
    uint64_t nz;
    uint64_t hs;
};

/*
 * returns kernel execution time
 * first arg is matrix in whatever format (explicit cast needed)
 * and second arg are matrix args (explicit cast needed)
 */
typedef double (*kernel_fp)(const void*, const void*);
typedef enum mult_datatype mult_datatype;
typedef enum matrix_format matrix_format;
typedef enum runner_type runner_type;

struct kernel_info {
    kernel_fp kernel;                        /* kernel function pointer, needed */
    matrix_format format;                    /* matrix format, needed */
    mult_datatype multiply_datatype;         /* multiplication datatype, data collection */
    runner_type runner;                      /* runner (host or device), data collection*/
    uint32_t cpu_mt_numthreads;              /* self-explainatory, data collection */
};

/*
 * after the final kernel, NULL
 */
void register_kernels_to_execute(const struct kernel_info *kerninfos);
void run_executor(int argc, char** argv);

#endif