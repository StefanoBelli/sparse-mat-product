#include<config.h>
#include<matrix/matrix-market/mmio.h>

int main(int argc, char** argv) {
    struct opened_mtx_file_list *head = setup(argc, argv);
    struct opened_mtx_file_list *bakhead = head;

    int i = 0;

    while(head) {
        MM_typecode matcode;

        if (mm_read_banner(head->fp, &matcode) != 0)
        {
            printf("unable to read banner for %s\n", head->name);
        } else if (
            mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) &&  //base checks for file interp - handle zeroes
            (mm_is_pattern(matcode) || mm_is_real(matcode)) && // determine format string - handle patterns
            (mm_is_symmetric(matcode) || mm_is_general(matcode))) // do I have to deal with symmetric mtxs?
        {
            i++;
            printf("%s is matching\n", head->name);
            //int m, n, nz;
            //mm_read_mtx_crd_size(head->fp, &m, &n, &nz);
            //printf(" m=%d rows, n=%d cols, nz=%d nonzeroes\n", m, n, nz);
        }

        head = head->next;
    }

    printf("--> %d mtxs are matching\n", i);

    free_all_opened_mtxs(&bakhead);
}