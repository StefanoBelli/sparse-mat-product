#ifndef FORMAT_MATRIX_H
#define FORMAT_MATRIX_H

#include "market-read.h"

struct csr_format {
    uint64_t *ja;
    uint64_t *irp;
    double *as;
};

void coo_to_csr(struct csr_format* out, const struct coo_format *coo, uint64_t nz, uint64_t m);
void free_csr_format(struct csr_format*);

struct ellpack_format {
    uint64_t **ja;
    double **as;
    uint64_t maxnz;
};

struct hll_format {
    uint64_t numblks;
    struct ellpack_format *blks;
};

void coo_to_hll(struct hll_format* out, const struct coo_format *coo, uint64_t nz, uint64_t m, uint64_t hs);
void free_hll_format(struct hll_format*, uint64_t hs);

void transpose_hll(struct hll_format* out, const struct hll_format* in, uint64_t hs);
void free_transposed_hll_format(struct hll_format*);

#endif