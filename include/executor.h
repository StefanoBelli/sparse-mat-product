#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <stdint.h>

// function ptr
// mult datatype (INT32, INT64, FLOAT32, FLOAT64)
// repr type (CSR, HLL)
// device / host type (CPU_SERIAL, CPU_MT, GPU)
// num of threads (relevant if device/host type == CPU_MT)

enum mult_datatype {
    FLOAT64,
    FLOAT32,
    INT32,
    INT64
};

enum matrix_repr {
    CSR,
    HLL
};

enum processor_type {
    CPU_SERIAL,
    CPU_MT,
    GPU
};

typedef double (*kernel_fp)(const void*, const void*);
typedef enum mult_datatype mult_datatype;
typedef enum matrix_repr matrix_repr;
typedef enum processor_type processor_type;

struct kernel_info {
    kernel_fp kernel;
    mult_datatype datatype;
    matrix_repr repr;
    processor_type proc;
    uint32_t nthrs;
};

void register_kernels_to_execute(uint32_t nk, const struct kernel_info *kerninfos);
void run_executor();

#endif