#include<config.h>
#include<matrix/format.h>
#include<matrix/market-read.h>
#include<utils.h>

#pragma GCC push_options
#pragma GCC optimize("O0")
int main(int argc, char** argv) {
    struct opened_mtx_file_list *head = setup(argc, argv);
    struct opened_mtx_file_list *bakhead = head;

    while(head) {
        uint64_t m = 0;
        uint64_t n = 0;
        uint64_t nz = 0;
        struct coo_format* mtx = NULL;
        if((mtx=read_matrix_market(head->fp, &m, &n, &nz))) {
            struct csr_format csr;
            coo_to_csr(&csr, mtx, nz, m);

            struct hll_format hll;
            coo_to_hll(&hll, mtx, nz, m, 32);

            free_csr_format(&csr);
            free_hll_format(&hll, 32);
            free_reset_ptr(mtx);
        }

        head = head->next;
    }

    free_all_opened_mtxs(&bakhead);
}
#pragma GCC pop_options