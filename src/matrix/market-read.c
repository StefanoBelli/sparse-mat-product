#include <stdio.h>
#include <string.h>
#include <utils.h>
#include <matrix/market-read.h>
#include <matrix/matrix-market/mmio.h>

#define __mm_is_valid(m) ( \
    mm_is_matrix(m) && \
    mm_is_sparse(m) && \
    (mm_is_real(m) || mm_is_pattern(m)) && \
    (mm_is_symmetric(m) || mm_is_general(m)) \
)

struct matrix_nonzero *read_matrix_market(FILE* fp, uint64_t *m, uint64_t *n, uint64_t *nz) {
    fseek(fp, 0, SEEK_SET);

    MM_typecode matcode;
    
    if(mm_read_banner(fp, &matcode) != 0) {
        return NULL;
    }

    if(!mm_is_valid(matcode)) {
        return NULL;
    }

    uint64_t tmp_nz = 0;
    if (mm_read_mtx_crd_size(fp, (int*) m, (int*) n, (int*) &tmp_nz)) {
        return NULL;
    }

    uint64_t initial_alloc_size = mm_is_symmetric(matcode) ? (tmp_nz << (uint64_t) 1) : tmp_nz;

    puts("calloc");
    struct matrix_nonzero *mtx = checked_calloc(struct matrix_nonzero, initial_alloc_size);

    const char *fmt = mm_is_real(matcode) ? "%d %d %lg\n" : "%d %d\n";

    uint64_t diag_nz = 0;

    for (uint64_t i_nz = 0; i_nz < tmp_nz; i_nz++) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
        fscanf(fp, fmt, &mtx[i_nz].i, &mtx[i_nz].j, &mtx[i_nz].val);
#pragma GCC diagnostic pop

        if (mm_is_pattern(matcode)) {
            mtx[i_nz].val = 1;
        }

        if (mm_is_symmetric(matcode)) {
            if (mtx[i_nz].i == mtx[i_nz].j) {
                diag_nz += 1;
            } else {
                uint64_t i_nz_shift = i_nz + tmp_nz;
                mtx[i_nz_shift].i = mtx[i_nz].j;
                mtx[i_nz_shift].j = mtx[i_nz].i;
                mtx[i_nz_shift].val = mtx[i_nz].val;
            }
        }
    }

    if(mm_is_symmetric(matcode)) {
        *nz = (tmp_nz << (uint64_t) 1) - diag_nz;
        mtx = checked_realloc(mtx, struct matrix_nonzero, *nz);
    } else {
        *nz = tmp_nz;
    }

    return mtx;
}