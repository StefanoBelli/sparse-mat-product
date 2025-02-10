#include <utils.h>
#include <config.h>
#include <executor.h>
#include <matrix/format.h>
#include <linux/limits.h>

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

    if(args->runner == CPU_MT) {
        uint32_t real_num_threads = kexinfo->cpu_mt_numthreads;
        if(real_num_threads == 0) {
            puts(" !! runner is CPU multithreaded, but num threads is set to 0");
            puts(" !! defaulting to 1");
            real_num_threads = 1;
        }

        if(args->set_num_thread) {
            if(args->set_num_thread(real_num_threads)) {
                puts(" ** this is unusual: unable to set num threads");
                puts(" ** terminating now, need to investigate...");
                exit(EXIT_FAILURE);
            }
        } else {
            puts(" !! runner is CPU multithreaded, but no set_num_thread function");
            puts(" !! pointer provided -- will not set");
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

static FILE* checked_fopen_writetrunc(const char* name) {
    FILE *fp = fopen(name, "w");
    if(fp == NULL) {
        log_error(fopen);
        exit(EXIT_FAILURE);
    }

    return fp;
}

static char* get_matrix_name(const struct opened_mtx_file_list *llist) {
    char *_ptr = strrchr(llist->name, '.');
    if(_ptr == NULL) {
        printf(" ** unable to get matrix name for %s... something is wrong", llist->name);
        puts(" ** terminating now, need to investigate...");
        exit(EXIT_FAILURE);
    }

    *_ptr = 0;
    return _ptr;
}

#define restore_matrix_name(ptr) \
    do { \
        *ptr = '.'; \
    } while(0)

#define fprintf_force_flush(_fp, _fmt, ...) \
    do { \
        fprintf(_fp, _fmt, __VA_ARGS__); \
        fflush(_fp); \
    } while(0)

#define fputs_force_flush(_fp, _str) \
    do { \
        fputs(_str, _fp); \
        fflush(_fp); \
    } while(0)

#define write_nz_csv_entry() \
    do { \
        char *ptr = get_matrix_name(head); \
        fprintf_force_flush(nzcsvfp, "%s,%ld\n", head->name, coo_p.nz); \
        restore_matrix_name(ptr); \
    } while(0)

#define mkresultsdir() mkcachedir("results")

static const char*
runner_to_string(runner_type runner) {
    const char* runner_as_str;
    if(runner == CPU_SERIAL) {
        runner_as_str = "serial";
    } else if(runner == CPU_MT) {
        runner_as_str = "cpu_mt";
    } else {
        runner_as_str = "gpu";
    }

    return runner_as_str;
}

static FILE 
*open_mtxresfile_and_write_csv_header(
        const struct opened_mtx_file_list *llist, 
        runner_type runner) {

    char filename_buf[PATH_MAX];
    char *ptr = get_matrix_name(llist);
    snprintf(filename_buf, PATH_MAX, "results/%s_%s.csv", llist->name, runner_to_string(runner));
    restore_matrix_name(ptr);
    FILE *mtxresfp = checked_fopen_writetrunc(filename_buf);
    fputs_force_flush(mtxresfp, "id,format,hack_size,cpu_mt_nthreads,time_it_took\n");

    return mtxresfp;
}

static void 
write_result_csv_entry(
        FILE *fp,
        int id, 
        double time, 
        const struct kernel_execution_info *kexinfo, 
        runner_type runner,
        const union format_args *fargs) {

    int real_num_threads = kexinfo->cpu_mt_numthreads;
    if(runner == CPU_MT && real_num_threads == 0) {
        real_num_threads = 1;
    }

    fprintf_force_flush(fp, "%d,%s,%d,%d,%lg\n", 
                            id, 
                            kexinfo->format == CSR ? "csr" : "hll",
                            kexinfo->format == HLL ? fargs->hll.hs : 0,
                            runner == CPU_MT ? real_num_threads : 0,
                            time);
}

void run_executor(int argc, char **argv, const struct executor_args *exe_args) {
    mkresultsdir();

    int num_trials;
    struct opened_mtx_file_list *head = setup(argc, argv, &num_trials);
    struct opened_mtx_file_list *bakhead = head;

    FILE *nzcsvfp = checked_fopen_writetrunc("results/nonzeroes.csv");
    fputs_force_flush(nzcsvfp, "matrix_name,non_zeroes\n");

    printf(
        " ** starting executions (%s)...\n"
        " ** use \"tail -f results/<filename>.csv\" to see results in real time\n", 
        runner_to_string(exe_args->runner));

    while(head) {
        printf(" ** loading %s\n", head->name);

        struct coo_params coo_p;
        coo_p.m = 0;
        coo_p.n = 0;
        coo_p.nz = 0;
        coo_p.coo = read_matrix_market(head->fp, &coo_p.m, &coo_p.n, &coo_p.nz);
        if(coo_p.coo != NULL) {
            printf(
                    " \t++ loaded %s: m = %ld, n = %ld, nz = %ld\n"
                    " \t!! running kernels on this matrix...\n", 
                head->name, coo_p.m, coo_p.n, coo_p.nz);

            write_nz_csv_entry();

            FILE *mtxresfp = open_mtxresfile_and_write_csv_header(head, exe_args->runner);
            int global_id = 1;
            for(int k = 0; k < exe_args->nkexs; k++) {
                union format_args args;
                void *fmt_mtx = prepare_kernel_args(&exe_args->kexinfos[k], &args, &coo_p);

                set_num_threads_before_kernel_execution(exe_args, &exe_args->kexinfos[k]);

                double *times = checked_malloc(double, num_trials);

                for(int i = 0; i < num_trials; i++) {
                    times[i] = exe_args->kexinfos[k].kernel(fmt_mtx, &args);
                    write_result_csv_entry(mtxresfp, global_id++, times[i], 
                                        &exe_args->kexinfos[k], exe_args->runner,
                                        &args);
                }

                free_reset_ptr(times);
                free_format_mtx(&exe_args->kexinfos[k], fmt_mtx, &args);
            }

            fclose(mtxresfp);
            free_reset_ptr(coo_p.coo);
        } else {
            printf(" \t!! unable to load %s\n", head->name);
        }

        head = head->next;
    }

    fclose(nzcsvfp);
    free_all_opened_mtxs(&bakhead);
}

#undef mkresultsdir
#undef fprintf_force_flush
#undef fputs_force_flush
#undef write_nz_csv_entry
#undef restore_matrix_name