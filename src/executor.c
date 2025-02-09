#include <utils.h>
#include <config.h>
#include <executor.h>
#include <matrix/format.h>

struct coo_params {
    struct coo_format *coo;
    uint64_t m;
    uint64_t n;
    uint64_t nz;
};

static void 
*prepare_kernel_args(
        const struct kernel_execution_info *kexinfo, 
        union format_args *args, 
        const struct coo_params* coo) {

    void *formatted_mtx;

    if (kexinfo->format == HLL) {
        args->hll.m = coo->m;
        args->hll.n = coo->n;
        args->hll.nz = coo->nz;

        if (kexinfo->hll_hack_size == 0) {
            args->hll.hs = 32;
        } else {
            args->hll.hs = kexinfo->hll_hack_size;
        }

        formatted_mtx = (void *)checked_malloc(struct hll_format, 1);
        coo_to_hll(formatted_mtx, coo->coo, coo->nz, coo->m, args->hll.hs);
    } else {
        args->csr.m = coo->m;
        args->csr.n = coo->n;
        args->csr.nz = coo->nz;

        formatted_mtx = (void *)checked_malloc(struct csr_format, 1);
        coo_to_csr(formatted_mtx, coo->coo, coo->nz, coo->m);
    }

    return formatted_mtx;
}

static void 
set_num_threads_before_kernel_execution(
        const struct executor_args *args,
        const struct kernel_execution_info *kexinfo) {

    if(args->runner == CPU_MT && args->set_num_thread != NULL) {
        if(args->set_num_thread(kexinfo->cpu_mt_numthreads)) {
            puts(" ** this is unusual: unable to set num threads");
            puts(" ** terminating now, need to investigate...");
            exit(EXIT_FAILURE);
        }
    }
}

static void 
free_format_mtx(
        const struct kernel_execution_info *kexinfo, 
        void *formatted_mtx, 
        const union format_args *fargs) {

    if (kexinfo->format == CSR) {
        free_csr_format((struct csr_format *)formatted_mtx);
    } else {
        free_hll_format((struct hll_format *)formatted_mtx, fargs->hll.hs);
    }

    free_reset_ptr(formatted_mtx);
}

void run_executor(int argc, char **argv, const struct executor_args *exe_args) {
    int num_trials;
    struct opened_mtx_file_list *head = setup(argc, argv, &num_trials);
    struct opened_mtx_file_list *bakhead = head;

    while(head) {
        struct coo_params coo_p;
        coo_p.m = 0;
        coo_p.n = 0;
        coo_p.nz = 0;
        coo_p.coo = read_matrix_market(head->fp, &coo_p.m, &coo_p.n, &coo_p.nz);

        if(coo_p.coo != NULL) {
            for(int k = 0; k < exe_args->nkexs; k++) {
                union format_args args;
                void *fmt_mtx = prepare_kernel_args(&exe_args->kexinfos[k], &args, &coo_p);

                set_num_threads_before_kernel_execution(exe_args, &exe_args->kexinfos[k]);

                double *times = checked_malloc(double, num_trials);

                for(int i = 0; i < num_trials; i++) {
                    times[i] = exe_args->kexinfos[k].kernel(fmt_mtx, &args);
                }

                free_reset_ptr(times);

                free_format_mtx(&exe_args->kexinfos[k], fmt_mtx, &args);
            }

            free_reset_ptr(coo_p.coo);
        }

        head = head->next;
    }

    free_all_opened_mtxs(&bakhead);
}