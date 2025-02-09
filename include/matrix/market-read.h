#ifndef MARKET_READ_H
#define MARKET_READ_H

#include <stdio.h>
#include <stdint.h>

struct coo_format {
    double v;
    uint64_t i;
    uint64_t j;
};

struct coo_format *read_matrix_market(FILE *fp, uint64_t *m, uint64_t *n, uint64_t *nz);

#endif