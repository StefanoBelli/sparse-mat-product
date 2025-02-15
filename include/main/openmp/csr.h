#ifndef CSR_OMP_H
#define CSR_OMP_H

#include <stdint.h>

void __kernel_csr(
        const uint64_t*, 
        const uint64_t*, 
        const double*, 
        uint32_t, 
        const double*, 
        double*);

#endif