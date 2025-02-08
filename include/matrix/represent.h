#ifndef REPRESENT_MATRIX_H
#define REPRESENT_MATRIX_H

#include "market-read.h"

struct csr_repr {
    uint64_t *ja;
    uint64_t *irp;
    double *as;
};

void to_csr(struct csr_repr* out, const struct coo_repr *coo, uint64_t nz, uint64_t m);
void free_csr_repr(struct csr_repr*);

struct ellpack_repr {
    uint64_t **ja;
    double **as;
    uint64_t maxnz;
    uint64_t m;
};

struct hll_repr {
    uint64_t numblks;
    struct ellpack_repr *blks;
};

void to_hll(struct hll_repr* out, const struct coo_repr *coo, uint64_t m, uint64_t hs);
void free_hll_repr(struct hll_repr*);

#endif