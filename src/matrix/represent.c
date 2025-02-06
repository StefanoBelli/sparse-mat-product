#include <math.h>
#include <utils.h>
#include <matrix/represent.h>

void to_csr(
        struct csr_repr *out, 
        const struct matrix_nonzero *coo, 
        uint64_t m, 
        uint64_t nz) {

    out->as = checked_calloc(double, nz);
    out->ja = checked_calloc(uint64_t, nz);
    out->irp = checked_calloc(uint64_t, m + 1);

    uint64_t irp_index = 0;
    uint64_t cur_row_index = 0;

    out->irp[irp_index++] = cur_row_index;

    for(uint64_t r = 0; r < m; r++) {
        for(uint64_t i = 0; i < nz; i++) {
            if(coo[i].i == r) {
                out->as[cur_row_index] = coo[i].val;
                out->ja[cur_row_index] = coo[i].j;
                cur_row_index++;
            }
        }

        out->irp[irp_index++] = cur_row_index;
    }

    out->irp = checked_realloc(out->irp, uint64_t, irp_index);
}

void free_csr_repr(struct csr_repr *cr) {
    free_reset_ptr(cr->as);
    free_reset_ptr(cr->ja);
    free_reset_ptr(cr->irp);
}

#define max(arr, arrsiz) ({ \
    uint64_t max_elem = 0; \
    for(uint64_t i = 0; i < arrsiz; i++) { \
        if(arr[i] > max_elem) { \
            max_elem = arr[i]; \
        } \
    } \
    max_elem; \
})

#define checked_matrix_calloc(ty, nr, nc) ({ \
    ty** _m = checked_calloc(ty*, nr); \
    for(uint64_t i = 0; i < nr; i++) { \
        _m[i] = checked_calloc(ty, nc); \
    } \
    _m; \
})

static void to_ellpack(
        struct ellpack_repr *out, 
        const struct matrix_nonzero *coo, 
        uint64_t nz, 
        uint64_t start_m, 
        uint64_t end_m) {

    out->m = end_m - start_m;
    uint64_t *nzs = checked_calloc(uint64_t, out->m);

    for(uint64_t i = 0; i < nz; i++) {
        if(start_m <= coo[i].i && coo[i].i < end_m) {
            nzs[coo[i].i - start_m]++;
        }
    }

    out->maxnz = max(nzs, out->m);
    free_reset_ptr(nzs);

    out->ja = checked_matrix_calloc(uint64_t, out->m, out->maxnz);
    out->as = checked_matrix_calloc(double, out->m, out->maxnz);

    uint64_t* cols = checked_calloc(uint64_t, out->m);

    for(uint64_t i = 0; i < nz; i++) {
        if(start_m <= coo[i].i && coo[i].i < end_m) {
            uint64_t eff_idx = coo[i].i - start_m;
            out->as[eff_idx][cols[eff_idx]] = coo[i].val;
            out->ja[eff_idx][cols[eff_idx]] = coo[i].j;
            cols[eff_idx]++;
        }
    }
    
    free_reset_ptr(cols);

    for(uint64_t i = 0; i < out->m; i++) {
        for(uint64_t j = 0; j < out->maxnz; j++) {
            if (j > 0 && out->as[i][j] == 0) {
                uint64_t remnz = out->maxnz - j;
                for (uint64_t k = 0; k < remnz; k++) {
                    out->ja[i][j + k] = out->ja[i][j - 1];
                }

                break;
            }
        }
    }
}

#undef checked_matrix_calloc
#undef max

static void free_ellpack_repr(struct ellpack_repr *er) {
    for(uint64_t i = 0; i < er->m; i++) {
        free_reset_ptr(er->ja[i]);
        free_reset_ptr(er->as[i]);
    }

    free_reset_ptr(er->ja);
    free_reset_ptr(er->as);
}

void to_hll(
        struct hll_repr *out, 
        const struct matrix_nonzero *coo, 
        uint64_t m, 
        uint64_t nz, 
        uint64_t hs) {

    uint64_t curblk = 0;

    out->numblks = ceil(m / hs);
    out->blks = checked_malloc(struct ellpack_repr, out->numblks);
    out->blksizs = checked_malloc(uint64_t, out->numblks);

    for (uint64_t r = 0; r < m; r += hs) {
        uint64_t final_row = r + hs;
        if(final_row > m) {
            final_row = m;
        }
        to_ellpack(&out->blks[curblk], coo, nz, r, final_row);
        out->blksizs[curblk] = final_row - r;
        out->numblks++; 
    }
}

void free_hll_repr(struct hll_repr *hr) {
    free_ellpack_repr(hr->blks);
    free_reset_ptr(hr->blks);
    free_reset_ptr(hr->blksizs);
}