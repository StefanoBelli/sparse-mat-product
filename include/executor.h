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
    uint64_t n;
    uint64_t nz;
};

struct hll_args {
    uint64_t m;
    uint64_t n;
    uint64_t nz;
    uint32_t hs;
};

union format_args {
    struct csr_args csr;
    struct hll_args hll;
};

/*
 * returns kernel execution time
 * first arg is matrix in whatever format (explicit cast needed)
 * and second arg are matrix args (explicit cast needed)
 */
typedef double (*kernel_fp)(const void*, const union format_args*, const char*);
typedef enum mult_datatype mult_datatype;
typedef enum matrix_format matrix_format;
typedef enum runner_type runner_type;

struct kernel_execution_info {
    kernel_fp kernel;                        /* kernel function pointer, needed */
    matrix_format format;                    /* matrix format, needed */
    uint32_t cpu_mt_numthreads;              /* self-explainatory, data collection, set nthreads when cpu_mt */
    uint32_t hll_hack_size;                  /* self-explainatory, data collection, set hacksize when hll */
    mult_datatype multiply_datatype;         /* multiplication datatype, data collection */
};

/* 
 * set num threads -- avoid the need to use -fopenmp on every translation unit or
 * complicate the Makefile. Client code sets this function pointer that is called
 * when needed. Must return whether thread set successful or not (1 or 0 respectfully).
 */
typedef int (*setnumthrs_fp)(uint32_t);

struct executor_args {
    setnumthrs_fp set_num_thread;
    const struct kernel_execution_info *kexinfos;
    int nkexs;
    runner_type runner;
};

/*
 * runner stored for data collection
 * 
 * for each kernel_execution
 *  for each matrix
 *   for each range(NTimes)
 *     time_it_took = kernel(mtx_in_format, format_params)
 *   done
 *  done
 * done
 */
void run_executor(int argc, char **argv, const struct executor_args *exesetup);

#endif