#include <matrix/market-read.h>
#include <matrix/represent.h>

#define ASSIGN_COO(_i, _j, _v, _idx) \
    do { \
        coo[_idx].i = _i; \
        coo[_idx].j = _j; \
        coo[_idx].v = _v; \
    } while(0)

void ellprod(double* y, uint64_t m, const struct ellpack_repr* ell, const double* x) {
    for(uint64_t i = 0; i < m; i++) {
        double t = 0;
        for(uint64_t j = 0; j < ell->maxnz; j++) {
            t += ell->as[i][j] * x[ell->ja[i][j]];
        }
        y[i] = t;
    }
}

#include<stdlib.h>

int main(int ac, char** argv) {

    //printf("cmdline = %s %s %s %s %s\n", argv[1], argv[2], argv[3], argv[4], argv[5]);

    uint64_t hs = atoi(argv[1]);
    double vek[] = {atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5])};
    double y[] = {0,0,0,0};

    uint64_t M = 4;
    uint64_t nz = 7;
    struct coo_repr coo[7];

    ASSIGN_COO(0, 0, 11, 0);
    ASSIGN_COO(0, 1, 12, 1);
    ASSIGN_COO(1, 1, 22, 2);
    ASSIGN_COO(1, 2, 23, 3);
    ASSIGN_COO(2, 2, 33, 4);
    ASSIGN_COO(3, 2, 43, 5);
    ASSIGN_COO(3, 3, 44, 6);

    struct csr_repr csr;

    coo_to_csr(&csr, coo, nz, M);

    printf(" csr ");
    for(uint64_t i = 0; i < M; i++) {
        double t = 0;

        for(uint64_t j = csr.irp[i]; j < csr.irp[i + 1]; j++) {
            t += csr.as[j] * vek[csr.ja[j]];
        }

        y[i] = t;
    }

    for(uint64_t i = 0; i < 4; i++) {
        printf("%lg,", y[i]);
    }
    printf("\n");

    free_csr_repr(&csr);

    y[0] = 0;
    y[1] = 0;
    y[2] = 0;
    y[3] = 0;

    struct hll_repr hll;
    coo_to_hll(&hll, coo, nz, M, hs);

    for(uint64_t numblk = 0; numblk < hll.numblks; numblk++) {
        double tmp[4];
        ellprod(tmp, hs, &hll.blks[numblk], vek);
        uint64_t endrow = hs;
        if (numblk * hs + hs > M) {
            endrow = M - hs;
        }

        for(uint64_t i = 0; i < endrow; i++) {
            y[i + numblk * hs] = tmp[i];
        }
    }

    free_hll_repr(&hll, hs);

    printf(" hll ");
    for(uint64_t i = 0; i < 4; i++) {
        printf("%lg,", y[i]);
    }
    printf("\n");
    return 0;
}