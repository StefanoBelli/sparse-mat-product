#include <matrix/market-read.h>
#include <matrix/represent.h>

#define ASSIGN_COO(_i, _j, _v, _idx) \
    do { \
        coo[_idx].i = _i; \
        coo[_idx].j = _j; \
        coo[_idx].v = _v; \
    } while(0)

void ellprod(char* y, int m, const struct ellpack_repr* ell, const double* x) {
    for(int i = 0; i < m; i++) {
        double t = 0;
        for(int j = 0; j < ell->maxnz; j++) {
            t += ell->as[i][j] * x[ell->ja[i][j]];
        }
        y[i] = t;
    }
}

int main() {

    int hs = 3;
    double vek[] = {1,0,1,0};
    double y[] = {0,0,0,0};

    int M = 4;
    int nz = 7;
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

    for(int i = 0; i < M; i++) {
        double t = 0;

        for(int j = csr.irp[i]; j < csr.irp[i + 1]; j++) {
            t += csr.as[j] * vek[csr.ja[j]];
        }

        y[i] = t;
    }

    for(int i = 0; i < 4; i++) {
        printf(" %lg,", y[i]);
    }
    printf("\n");

    free_csr_repr(&csr);

    y[0] = 0;
    y[1] = 0;
    y[2] = 0;
    y[3] = 0;

    struct hll_repr hll;
    coo_to_hll(&hll, coo, nz, M, hs);

    for(int numblk = 0; numblk < hll.numblks; numblk++) {
        char tmp[4];
        ellprod(tmp, hs, &hll.blks[numblk], vek);
        int endrow = hs;
        if (numblk * hs + hs > M) {
            endrow = M - hs;
        }

        for(int i = 0; i < endrow; i++) {
            y[i + numblk * hs] = tmp[i];
        }
    }

    free_hll_repr(&hll, hs);

    for(int i = 0; i < 4; i++) {
        printf(" %lg,", y[i]);
    }
    printf("\n");
    return 0;
}