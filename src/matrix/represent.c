#include <math.h>
#include <utils.h>
#include <matrix/represent.h>

void 
coo_to_csr(
        struct csr_repr* out, 
        const struct coo_repr *coo, 
        uint64_t nz, 
        uint64_t m) {

    out->as = checked_calloc(double, nz);
    out->ja = checked_calloc(uint64_t, nz);
    out->irp = checked_calloc(uint64_t, m + 1);

    uint64_t irp_index = 0;
    uint64_t cur_row_idx = coo[0].i;

    out->irp[irp_index++] = cur_row_idx;

    for(uint64_t i = 0; i < nz; i++) {
        if(cur_row_idx != coo[i].i) {
            cur_row_idx = coo[i].i;
            out->irp[irp_index++] = i;
        }

        out->as[i] = coo[i].v;
        out->ja[i] = coo[i].j;
    }

    out->irp[irp_index++] = nz;

    out->irp = checked_realloc(out->irp, uint64_t, irp_index);
}

void free_csr_repr(struct csr_repr* cr) {
    free_reset_ptr(cr->as);
    free_reset_ptr(cr->ja);
    free_reset_ptr(cr->irp);
}

#define my_max(arr, arrsiz) ({ \
    uint64_t max_elem = 0; \
    for(uint64_t i = 0; i < arrsiz; i++) { \
        if(arr[i] > max_elem) { \
            max_elem = arr[i]; \
        } \
    } \
    max_elem; \
})

#define checked_matrix_calloc(ty, m, n) ({ \
    ty** mtx = checked_calloc(ty*, m); \
    for(uint64_t i = 0; i < m; i++) { \
        mtx[i] = checked_calloc(ty, n); \
    } \
    mtx; \
})

static void 
coo_to_ellpack(
        struct ellpack_repr *out, 
        const struct coo_repr *coo, 
        uint64_t nz, 
        uint64_t m, 
        uint64_t s_row, 
        uint64_t s_idx_incl) {

    uint64_t *nzrs = checked_calloc(uint64_t, m);
    for(uint64_t i = 0; i < nz; i++) {
        nzrs[coo[i + s_idx_incl].i - s_row]++;
    }

    out->maxnz = my_max(nzrs, m);

    free_reset_ptr(nzrs);

    out->ja = checked_matrix_calloc(uint64_t, m, out->maxnz);
    out->as = checked_matrix_calloc(double, m, out->maxnz);

    uint64_t *cols = checked_calloc(uint64_t, m);

    for(uint64_t i = 0; i < nz; i++) {
        uint64_t eff_i = coo[i + s_idx_incl].i - s_row;
        out->as[eff_i][cols[eff_i]] = coo[i + s_idx_incl].v;
        out->ja[eff_i][cols[eff_i]] = coo[i + s_idx_incl].j;
        cols[eff_i]++;
    }

    free_reset_ptr(cols);

    for(uint64_t i = 0; i < m; i++) {
        uint64_t last_valid_index = 0;
        for(uint64_t j = 0; j < out->maxnz; j++) {
            if(out->as[i][j] != 0) {
                last_valid_index = out->ja[i][j];
            } else {
                out->ja[i][j] = last_valid_index;
            }
        }
    }
}

#undef my_max
#undef checked_matrix_calloc

static void free_ellpack_repr(struct ellpack_repr *er, uint64_t m) {
    for(uint64_t i = 0; i < m; i++) {
        free_reset_ptr(er->ja[i]);
        free_reset_ptr(er->as[i]);
    }

    free_reset_ptr(er->ja);
    free_reset_ptr(er->as);
    er->maxnz = 0;
}

#define u64cast(x) ((uint64_t)x)
#define dcast(x) ((double) x)

void 
coo_to_hll(
        struct hll_repr* out, 
        const struct coo_repr *coo, 
        uint64_t nz,
        uint64_t m,
        uint64_t hs) {

    double _m_d_hs = dcast(m) / dcast(hs);

    out->numblks = m % hs ? u64cast(floor(_m_d_hs)) + 1 : u64cast(_m_d_hs);
    out->blks = checked_malloc(struct ellpack_repr, out->numblks);

    uint64_t starting_coo_incl_idx = 0;
    uint64_t i_idx = 0;
    uint64_t blk_idx = 0;

    for(uint64_t r = 0; r < m; r += hs) {
        uint64_t final_row = r + hs;
        if(final_row > m) {
            final_row = m;
        }

        uint64_t hack_nz = 0;
        while(i_idx < nz && coo[i_idx].i < final_row) {
            hack_nz++;
            i_idx++;
        }

        coo_to_ellpack(&out->blks[blk_idx++], coo, hack_nz, hs, r, starting_coo_incl_idx);

        starting_coo_incl_idx += hack_nz;
    }
}

#undef u64cast
#undef dcast

void free_hll_repr(struct hll_repr* hr, uint64_t hs) {
    for(uint64_t i = 0; i < hr->numblks; i++) {
        free_ellpack_repr(&hr->blks[i], hs);
    }

    free_reset_ptr(hr->blks);
    hr->numblks = 0;
}