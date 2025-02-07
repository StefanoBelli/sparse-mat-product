#include<config.h>
#include<matrix/represent.h>
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
        struct matrix_nonzero* mtx = NULL;
        printf("%s\n", head->name);
        if((mtx=read_matrix_market(head->fp, &m, &n, &nz))) {
            printf("trying to access matrix %s\n\tm=%ld, n=%ld, nz=%ld\n", head->name, m, n, nz);
            //struct hll_repr hll;
            //to_hll(&hll, mtx, m, nz, 32);
            //free_hll_repr(&hll);
            free_reset_ptr(mtx);
        }

        head = head->next;
    }

    free_all_opened_mtxs(&bakhead);
}
#pragma GCC pop_options