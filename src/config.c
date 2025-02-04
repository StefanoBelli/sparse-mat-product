#define _POSIX_C_SOURCE 200809L

#include<unistd.h>
#include<getopt.h>
#include<stdio.h>
#include<string.h>
#include<errno.h>
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

void setup(int argc, char** argv) {
    struct program_config cfg;
    obtain_program_config(argc, argv, &cfg);

    errno = 0;
    if(mkcachedir(cfg.cachedir_path)) {
        log_error(mkdir);
        exit(EXIT_FAILURE);
    }

    struct tracking_files* tracked_files = add_file_to_track(NULL, 0);

    if(cfg.i_need_to_track_files) {
#ifdef FIX_BROKEN_CACHE
        char cachedir_abspath[PATH_MAX];
        char cwd_abspath[PATH_MAX];

        realpath(cfg.cachedir_path, cachedir_abspath);
        getcwd(cwd_abspath, PATH_MAX);

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
}