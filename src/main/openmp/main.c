#include<config.h>
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
        if((mtx=read_matrix_market(head->fp, &m, &n, &nz))) {
            printf("trying to access matrix %s\n\tm=%ld, n=%ld, nz=%ld\n", head->name, m, n, nz);
            uint64_t ui = 0;
            uint64_t uj = 0;
            double uval = 0;

            for(uint64_t c = 0; c < nz; c++) {
                int z1 = mtx[c].i;
                int z2 = mtx[c].j;
                double z3 = mtx[c].val;

                ui += z1;
                uj += z2;
                uval += z3;
            }

            printf("\t\t%ld, %ld, %lg (yesfree)\n", ui, uj, uval);

            free(mtx);
        }

        head = head->next;
    }

    free_all_opened_mtxs(&bakhead);
}
#pragma GCC pop_options