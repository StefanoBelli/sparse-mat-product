#include <stdio.h>
#include <string.h>
#include <utils.h>
#include <matrix/market-read.h>
#include <matrix/matrix-market/mmio.h>

#define __mmio_mm_is_valid(m) ( \
    mm_is_matrix(m) && \
    mm_is_sparse(m) && \
    (mm_is_real(m) || mm_is_pattern(m)) && \
    (mm_is_symmetric(m) || mm_is_general(m)) \
)

static int initialize_mmio_read(MM_typecode *typecode, FILE* f, uint64_t *m, uint64_t *n, uint64_t *nz) {
    if(mm_read_banner(f, typecode)) {
        return -1;
    }

    if(!__mmio_mm_is_valid(*typecode)) {
        return -1;
    }

    if (mm_read_mtx_crd_size(f, (int*) m, (int*) n, (int*) nz)) {
        return -1;
    }

    return 0;
}

static inline uint64_t symmetry_fixup(struct matrix_nonzero *m, uint64_t index, uint64_t orig_nz) {
    uint64_t diagonal_nonzeroes = 0;

    if (m[index].i == m[index].j) {
        diagonal_nonzeroes += 1;
    } else {
        uint64_t i_nz_shift = index + orig_nz;
        m[i_nz_shift].i = m[index].j;
        m[i_nz_shift].j = m[index].i;
        m[i_nz_shift].val = m[index].val;
    }

    return diagonal_nonzeroes;
}

struct matrix_nonzero *read_matrix_market(FILE* fp, uint64_t *m, uint64_t *n, uint64_t *nz) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    fseek(fp, 0, SEEK_SET);
#pragma GCC diagnostic pop

    MM_typecode matcode;
    uint64_t tmp_nz = 0;
    uint64_t diag_nz = 0;

    if(initialize_mmio_read(&matcode, fp, m, n, &tmp_nz)) {
        return NULL;
    }

    uint64_t initial_alloc_size = mm_is_symmetric(matcode) ? (tmp_nz * 2) : tmp_nz;
    struct matrix_nonzero *mtx = checked_calloc(struct matrix_nonzero, initial_alloc_size);
    const char *fmt = mm_is_real(matcode) ? "%d %d %lg\n" : "%d %d\n";

    for (uint64_t i_nz = 0; i_nz < tmp_nz; i_nz++) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
        fscanf(fp, fmt, &mtx[i_nz].i, &mtx[i_nz].j, &mtx[i_nz].val);
#pragma GCC diagnostic pop

        if (mm_is_pattern(matcode)) {
            mtx[i_nz].val = 1;
        }

        if (mm_is_symmetric(matcode)) {
            diag_nz += symmetry_fixup(mtx, i_nz, tmp_nz);
        }
    }

    if(mm_is_symmetric(matcode)) {
        *nz = (tmp_nz * 2) - diag_nz;
        mtx = checked_realloc(mtx, struct matrix_nonzero, *nz);
    } else {
        *nz = tmp_nz;
    }

    return mtx;
}