#ifndef HLL_OMP_H
#define HLL_OMP_H

#include <matrix/format.h>
#include <stdint.h>

void __kernel_hll(
        const struct ellpack_format *, 
        uint64_t, 
        uint32_t, 
        uint32_t, 
        const double *, 
        double *);

#endif