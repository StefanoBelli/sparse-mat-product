#define _POSIX_C_SOURCE 200809L

#include<unistd.h>
#include<getopt.h>
#include<stdio.h>
#include<string.h>
#include<errno.h>
#include<dirent.h>
#include<linux/limits.h>
#include<file-util/tracking/tracking.h>
#include<file-util/tracking/utils.h>
#include<utils.h>
#include<matrices.h>
#include<config.h>

#define num_default_matrices (sizeof(matrices) / sizeof(char*))

struct program_config {
    int i_need_to_track_files;
    char cachedir_path[PATH_MAX + 1];
};

static void add_files_to_track_from_file(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if(fp == NULL) {
        log_warn("fopen(%s): %s",filename, strerror(errno));
        return;
    }

    char* lineptr = NULL;
    size_t sz;

    while(getline(&lineptr, &sz, fp) != -1) {
        add_file_to_track(lineptr, num_default_matrices);
        free_reset_ptr(lineptr);
    }
    
    free_reset_ptr(lineptr);

    fclose(fp);
}

static void get_default_cachedir(char *buf) {
    char* base = getenv("HOME");
    snprintf(buf, PATH_MAX + 1, "%s/.scpa2425_cachedir_sparse_mat_prod", base ? base : ".");
}

static void enable_tracking_of_mandatory_matrices() {
    for(size_t i = 0; i < num_default_matrices; i++) {
        add_file_to_track(matrices[i], num_default_matrices);
    }
}

static void obtain_program_config(int argc, char** argv, struct program_config *cfg) {
    enable_tracking_of_mandatory_matrices();

    cfg->i_need_to_track_files = 1;
    get_default_cachedir(cfg->cachedir_path);

    int opt;
    while((opt = getopt(argc, argv, "dem:t:f:")) != -1) {
        switch(opt) {
            case 'd':
            cfg->i_need_to_track_files = 0; break;
            
            case 'e':
            cfg->i_need_to_track_files = 1; break;

            case 'm':
            snprintf(cfg->cachedir_path, PATH_MAX + 1, "%s", optarg); break;

            case 't':
            add_file_to_track(optarg, num_default_matrices); break;

            case 'f':
            add_files_to_track_from_file(optarg); break;
        }
    }
}

static struct opened_mtx_file_list* open_all_mtx_files_from_dir(const char* dirpath) {
    printf("searching for all .mtx files in \"%s\"...\n", dirpath);

    int num_mtxs_found = 0;
    DIR* d = opendir(dirpath);
    if(d == NULL) {
        log_error(opendir);
        exit(EXIT_FAILURE);
    }

    struct opened_mtx_file_list *head = NULL;
    struct opened_mtx_file_list *curr = NULL;

    struct dirent *dent;

    errno = 0;
    while((dent = readdir(d))) {
        if(!has_file_ext(dent->d_name, ".mtx")) {
            char pathbuf[PATH_MAX];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
            snprintf(pathbuf, PATH_MAX, "%s/%s", dirpath, dent->d_name);
#pragma GCC diagnostic pop

            FILE *curfp = fopen(pathbuf, "r");
            if(curfp == NULL) {
                log_error(fopen);
            } else {
                size_t dnamelen = strlen(dent->d_name);
                struct opened_mtx_file_list *newcurr = checked_malloc(struct opened_mtx_file_list, 1);
                newcurr->fp = curfp;
                newcurr->name = checked_malloc(char, dnamelen + 1);
                newcurr->name[dnamelen] = 0;
                memcpy(newcurr->name, dent->d_name, dnamelen);
                newcurr->next = NULL;
                num_mtxs_found++;

                if(head == NULL) {
                    head = newcurr;
                }

                if(curr == NULL) {
                    curr = newcurr;
                } else {
                    curr->next = newcurr;
                    curr = newcurr;
                }
            }
        }
        errno = 0;
    }

    if(errno) {
        log_error(readdir);
    }

    if(closedir(d)) {
        log_error(closedir);
    }

    printf("%d mtxs were found!\n", num_mtxs_found);

    return head;
}

void free_all_opened_mtxs(struct opened_mtx_file_list** head) {
    while(*head) {
        if(fclose((*head)->fp)) {
            log_error(fclose);
        }

        free_reset_ptr((*head)->name);

        struct opened_mtx_file_list *next = (*head)->next;
        free_reset_ptr(*head);

        *head = next;
    }
}

struct opened_mtx_file_list* setup(int argc, char** argv) {
    struct program_config cfg;
    obtain_program_config(argc, argv, &cfg);

    if(mkcachedir(cfg.cachedir_path)) {
        log_error(mkdir);
        exit(EXIT_FAILURE);
    }

    struct tracking_files* tracked_files = add_file_to_track(NULL, 0);

    if(cfg.i_need_to_track_files) {
#ifdef FIX_BROKEN_CACHE
        char cachedir_abspath[PATH_MAX];
        char cwd_abspath[PATH_MAX];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
        realpath(cfg.cachedir_path, cachedir_abspath);
        getcwd(cwd_abspath, PATH_MAX);
#pragma GCC diagnostic pop

        if(!strcmp(cwd_abspath, cachedir_abspath)) {
            puts(">>> file tracker may remove some files");
            puts(">>> by specifying this current working directory");
            puts(">>> you will loose them");
            puts(">>> specify another directory with \"-m <dir>\" or");
            puts(">>> use \"-d\" to disable file tracker");
            exit(EXIT_FAILURE);
        }
#endif

        track_files(cfg.cachedir_path, tracked_files);
    }
    
    free_tracking_files(&tracked_files);

    return open_all_mtx_files_from_dir(cfg.cachedir_path);
}