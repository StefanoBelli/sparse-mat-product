#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <stdint.h>

enum mult_datatype {
    FLOAT64,
    FLOAT32,
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
    uint32_t cpu_mt_numthreads;              /* self-explainatory, data collection */
};

/*
 * runner stored for data collection
 * kerninfo is NULL-terminated
 * 
 * for each kernel
 *  for each matrix
 *   for each range(NTimes)
 *     time_it_took = kernel(mtx_in_format, format_params)
 *   done
 *  done
 * done
 */
void run_executor(int argc, char **argv, runner_type runner, const struct kernel_info *kerninfos);

#endif