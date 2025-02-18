#include <main/openmp/hll.h>

void __kernel_hll(
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
                ell_tmp += ell.as[(i * ell.maxnz) + j] * x[ell.ja[(i * ell.maxnz) + j]];
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
