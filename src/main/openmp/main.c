#include<config.h>
#include<matrix/matrix-market/mmio.h>
#include<sys/time.h>
#include<utils.h>
#define expand(e) e
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
            int m, n, nz;
            printf("%d\n", mm_read_mtx_crd_size(head->fp, &m, &n, &nz));
            printf(" m=%d rows, n=%d cols, nz=%d nonzeroes\n", m, n, nz);

            int* row_indexes = checked_malloc(int, nz);
            int* col_indexes = checked_malloc(int, nz);
            double* values = checked_malloc(double, nz);

            for(int j = 0; j < nz; j++) {
                const char* fmt = mm_is_pattern(matcode) ? "%d %d\n" : "%d %d %lg\n";
                int mi, mj;
                double mval;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
                fscanf(head->fp, fmt, &mi, &mj, &mval);
#pragma GCC diagnostic pop
                row_indexes[j] = mi - 1;
                col_indexes[j] = mj - 1;
                values[j] = mm_is_pattern(matcode) ? 1 : mval;

                //printf("\t%s (%d) --> mi = %d, mj = %d, mval = %lg\n", head->name, j, mi, mj, mm_is_pattern(matcode) ? 1.0 : mval);
            }

            int global_counter = 0;
            for(int r = 0; r < m; r++) {
                for(int c = 0; c < n; c++) {
                    /*printf("%s i = %d, j = %d, val = ", head->name, r, c);
                    if(row_indexes[global_counter % nz] == r && col_indexes[global_counter] == c) {
                        printf("%lg\n", values[global_counter]);
                    } else {
                        puts("0");
                    }*/
                   printf("%d %d %lg\n", row_indexes[global_counter % nz], col_indexes[global_counter % nz], values[global_counter % nz]);
                    global_counter++;
                    //usleep(10000);
                }
            }
            free_reset_ptr(row_indexes);
            free_reset_ptr(col_indexes);
            free_reset_ptr(values);
        }

        head = head->next;
    }

    printf("--> %d mtxs are matching\n", i);

    free_all_opened_mtxs(&bakhead);
}