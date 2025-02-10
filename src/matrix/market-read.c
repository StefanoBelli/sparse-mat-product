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

static inline void symmetry_fixup(struct coo_format *m, uint64_t index, uint64_t orig_nz, uint64_t *dnz, uint64_t *ez) {
    if (m[index].i == m[index].j) {
        if(m[index].v != 0) {
            *dnz += 1;
        } else {
            *ez += 1;
        }
    } else {
        uint64_t i_nz_shift = index + orig_nz;
        m[i_nz_shift].i = m[index].j;
        m[i_nz_shift].j = m[index].i;
        m[i_nz_shift].v = m[index].v;
        if(m[i_nz_shift].v == 0) {
            *ez += 1;
        }
    }
}

static int coo_sparse_comparator(const void *m1, const void *m2) {
    const struct coo_format *mtx1 = (const struct coo_format*) m1;
    const struct coo_format *mtx2 = (const struct coo_format*) m2;

    if(mtx1->v == 0 && mtx2->v == 0) {
        return 0;
    } else if(mtx1->v == 0 && mtx2->v != 0) {
        return 1;
    } else if(mtx1->v != 0 && mtx2->v == 0) {
        return -1;
    } else {
        if(mtx1->i > mtx2->i) {
            return 1;
        } else if(mtx1->i < mtx2->j) {
            return -1;
        } else {
            return 0;
        }
    }
}

struct coo_format *read_matrix_market(FILE* fp, uint64_t *m, uint64_t *n, uint64_t *nz) {

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
    struct coo_format *mtx = checked_calloc(struct coo_format, initial_alloc_size);
    const char *fmt = mm_is_real(matcode) ? "%d %d %lg\n" : "%d %d\n";

    uint64_t ez = 0;

    for (uint64_t i_nz = 0; i_nz < tmp_nz; i_nz++) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
        fscanf(fp, fmt, &mtx[i_nz].i, &mtx[i_nz].j, &mtx[i_nz].v);
#pragma GCC diagnostic pop

        mtx[i_nz].i--;
        mtx[i_nz].j--;

        if (mm_is_pattern(matcode)) {
            mtx[i_nz].v = 1;
        } else {
            if(mtx[i_nz].v == 0) {
                ez += 1;
            }
        }

        if (mm_is_symmetric(matcode)) {
            symmetry_fixup(mtx, i_nz, tmp_nz, &diag_nz, &ez);
        }
    }

    if(mm_is_symmetric(matcode)) {
        *nz = (tmp_nz * 2) - diag_nz;
    } else {
        *nz = tmp_nz;
    }

    *nz -= ez;

    qsort(mtx, initial_alloc_size, sizeof(struct coo_format), coo_sparse_comparator);

    if(*nz < initial_alloc_size) {
        mtx = checked_realloc(mtx, struct coo_format, *nz);
    }

    return mtx;
}