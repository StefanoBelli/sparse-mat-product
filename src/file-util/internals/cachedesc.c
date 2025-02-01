#define _POSIX_C_SOURCE 200809L

#include<errno.h>
#include<utils.h>
#include<string.h>
#include<sys/stat.h>
#include<file-util/internals/cachedesc.h>

extern int errno;

struct valid_line_list {
    char* lineptr;
    struct valid_line_list *next;
};

static int has_file_ext(const char* filename, const char* ext) {
    char *dotat = strrchr(filename, '.');
    if(dotat == NULL) {
        return 1;
    }

    return strcmp(dotat, ext);
}

static void replace_file_content(const char* filepath, FILE **fp, struct valid_line_list *vll_new_head) {
    *fp = freopen(filepath, "w", *fp);
    if(*fp == NULL) {
        abort();
    }

    while(vll_new_head) {
        fprintf(*fp, "%s\n", vll_new_head->lineptr);

        struct valid_line_list *bak = vll_new_head;

        vll_new_head = bak->next;

        free_reset_ptr(bak->lineptr);
        free_reset_ptr(bak);
    }

    fflush(*fp);

    *fp = freopen(filepath, "r", *fp);
}

#define check_failure(ptr, rv) \
    do { \
        if(ptr == NULL) { \
            int bak_errno = errno; \
            free(cd); \
            cd = NULL; \
            errno = bak_errno; \
            return rv; \
        } \
    } while(0)

int open_cachedir(const char *cachedir, struct cachedesc **cd_out) {
    struct cachedesc *cd = checked_malloc(struct cachedesc, 1);

    if(mkdir(cachedir, 0755) == -1 && errno != EEXIST) {
        free_reset_ptr(cd);
        return OPEN_CACHEDIR_MKDIR_FAIL;
    }

    DIR* dir = opendir(cachedir);
    check_failure(dir, OPEN_CACHEDIR_OPENDIR_FAIL);

    char cachedesc_filename[PATH_MAX + 1];
    memset(cachedesc_filename, 0, PATH_MAX + 1);
    snprintf(cachedesc_filename, PATH_MAX, "%s/cachedesc", cachedir);

    FILE* fp = fopen(cachedesc_filename, "a");
    check_failure(fp, OPEN_CACHEDIR_FOPEN_FAIL);
    
    fp = freopen(cachedesc_filename, "r", fp);
    check_failure(fp, OPEN_CACHEDIR_FOPEN_FAIL);

    memset(cd->cachedir_path, 0, PATH_MAX + 1);
    memcpy(cd->cachedir_path, cachedir, strlen(cachedir));

    memset(cd->cachedesc_path, 0, PATH_MAX + 1);
    memcpy(cd->cachedesc_path, cachedesc_filename, strlen(cachedesc_filename));

    cd->cachedir = dir;
    cd->fp = fp;
    *cd_out = cd;

    return 0;
}

#undef check_failure

void close_cachedir(const struct cachedesc *cd) {

}

#define add_valid_line(lnptr) \
    do { \
        struct valid_line_list* curln = checked_malloc(struct valid_line_list, 1); \
        curln->lineptr = lnptr; \
        curln->next = NULL; \
        if(vll_head == NULL) { \
            vll_head = curln; \
        } \
        if(vll_cur != NULL) { \
            vll_cur->next = curln; \
        } \
        vll_cur = curln; \
    } while(0)
        
void fix_broken_cachedesc(struct cachedesc *cd) {
    char *lineptr = NULL;
    size_t bufsiz = 0;
    struct valid_line_list *vll_cur = NULL;
    struct valid_line_list *vll_head = NULL;

    while(getline(&lineptr, &bufsiz, cd->fp) != -1) {
        int linelen = strlen(lineptr);
        if(lineptr[linelen - 1] == '\n') {
            lineptr[linelen - 1] = 0;
        }

        char* cached_filename = strtok(lineptr, " ");
        if(cached_filename != NULL) {
            char* cached_filename_endptr = cached_filename + strlen(cached_filename);
            char* cached_checksum = strtok(NULL, " ");
            if(cached_checksum != NULL) {
                char cached_abs_filepath[PATH_MAX + 1];
                memset(cached_abs_filepath, 0, PATH_MAX + 1);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
                snprintf(cached_abs_filepath, PATH_MAX, "%s/%s", 
                    cd->cachedir_path, cached_filename);
#pragma GCC diagnostic pop

                struct stat statbuf; 
                if(!stat(cached_abs_filepath, &statbuf) && S_ISREG(statbuf.st_mode)) {
                    if(!has_file_ext(cached_filename, ".mtx")) {
                        cached_filename_endptr[0] = ' ';
                        add_valid_line(lineptr);
                        goto next;
                    } else {
                        if(!has_file_ext(cached_filename, ".gz")) {
                            cached_filename_endptr[-3] = 0;
                            if(!has_file_ext(cached_filename, ".tar")) {
                                cached_filename_endptr[-3] = '.';
                                cached_filename_endptr[0] =  ' ';
                                add_valid_line(lineptr);
                                goto next;
                            }
                        }
                    } 
                }
            }     
        }

        free(lineptr);
next:
        lineptr = NULL;
    }

    replace_file_content(cd->cachedesc_path, &cd->fp, vll_head);
    vll_cur = NULL;
    vll_head = NULL;
}

#undef add_valid_line

void fix_broken_cache(const struct cachedesc *cd) {

}

void update_cachedesc_with_csum(const struct cachedesc *cd, const char* md5sum_stdout) {

}

void get_csum_from_cachedesc(const struct cachedesc *cd, const char* filename) {

}