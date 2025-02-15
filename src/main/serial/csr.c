#include <main/serial/csr.h>

void __kernel_csr(
        const uint64_t *irp, 
        const uint64_t *ja, 
        const double *as, 
        uint32_t m, 
        const double *x, 
        double *y) {

    for(uint64_t i = 0; i < m; i++) {
        double t = 0;
        for(uint64_t j = irp[i]; j < irp[i + 1]; j++) {
            t += as[j] * x[ja[j]];
        }
        y[i] = t;
    }
}