#define _POSIX_C_SOURCE 200809L

#include<dirent.h>
#include<utils.h>
#include<file-util/tracking/cachedesc.h>
#include<file-util/tracking/utils.h>

struct valid_line_list {
    char* lineptr;
    struct valid_line_list *next;
};

static void remove_directory_recursive(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) {
        log_error(opendir);
        return;
    }

    struct dirent *dent;

    errno = 0;
    while ((dent = readdir(dir))) {
        if (!strcmp(dent->d_name, ".") || !strcmp(dent->d_name, "..")) {
            continue;
        }
        
        char abspath[PATH_MAX + 1];
        memset(abspath, 0, PATH_MAX + 1);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
        snprintf(abspath, PATH_MAX, "%s/%s", path, dent->d_name);
#pragma GCC diagnostic pop
        printf("%s\n", abspath);

        struct stat statbuf;
        if (!stat(abspath, &statbuf)) {
            if (S_ISDIR(statbuf.st_mode)) {
                remove_directory_recursive(abspath);
            } else {
                if (remove(abspath) != 0) {
                    log_error(remove);
                }
            }
        } else {
            log_error(stat);
        }

        errno = 0;
    }

    if(errno) {
        log_error(readdir);
    }

    if(closedir(dir)) {
        log_error(closedir);
    }

    if(remove(path)) {
        log_error(remove);
    }
}

#define __freopen(__path, __mode, __fp) \
    do { \
        if((__fp = freopen(__path, __mode, __fp)) == NULL) { \
            log_error(freopen); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static void replace_file_content(const char* filepath, FILE **fp, struct valid_line_list *vll_new_head) {
    __freopen(filepath, "w", *fp);

    while(vll_new_head) {
        fprintf(*fp, "%s\n", vll_new_head->lineptr);

        struct valid_line_list *bak = vll_new_head;

        vll_new_head = bak->next;

        free_reset_ptr(bak->lineptr);
        free_reset_ptr(bak);
    }

    __freopen(filepath, "r+", *fp);
}

#define fail(fn) \
    do { \
        log_error(fn); \
        exit(EXIT_FAILURE); \
    } while(0)

void open_cachedir(const char *cachedir, struct cachedesc **cd_out) {
    struct cachedesc *cd = checked_malloc(struct cachedesc, 1);

    if(mkcachedir(cachedir)) {
        fail(mkdir);
    }

    DIR* dir = opendir(cachedir);
    if(dir == NULL) {
        fail(opendir);
    }

    char cachedesc_filename[PATH_MAX + 1];
    memset(cachedesc_filename, 0, PATH_MAX + 1);
    snprintf(cachedesc_filename, PATH_MAX, "%s/cachedesc", cachedir);

    FILE* fp = fopen(cachedesc_filename, "a");
    if(fp == NULL) {
        fail(fopen);
    }

    __freopen(cachedesc_filename, "r+", fp);

    memset(cd->cachedir_path, 0, PATH_MAX + 1);
    memcpy(cd->cachedir_path, cachedir, strlen(cachedir));

    memset(cd->cachedesc_path, 0, PATH_MAX + 1);
    memcpy(cd->cachedesc_path, cachedesc_filename, strlen(cachedesc_filename));

    cd->cachedir = dir;
    cd->fp = fp;
    *cd_out = cd;
}

#undef __freopen
#undef fail

void close_cachedir(struct cachedesc **cd) {
    if(closedir((*cd)->cachedir)) {
        log_error(closedir);
    }

    if(fclose((*cd)->fp)) {
        log_error(fclose);
    }

    free_reset_ptr(*cd);
}

#define add_valid_line() \
    do { \
        struct valid_line_list* curln = checked_malloc(struct valid_line_list, 1); \
        curln->lineptr = lineptr; \
        curln->next = NULL; \
        if(vll_head == NULL) { \
            vll_head = curln; \
        } \
        if(vll_cur != NULL) { \
            vll_cur->next = curln; \
        } \
        vll_cur = curln; \
    } while(0)

#define this_is_a_valid_line() \
    do { \
        cached_filename_endptr[0] = ' '; \
        add_valid_line(); \
        goto next; \
    } while(0)
        
void fix_broken_cachedesc(struct cachedesc *cd) {
    fseek(cd->fp, 0, SEEK_SET);

    char *lineptr = NULL;
    size_t bufsiz = 0;
    struct valid_line_list *vll_cur = NULL;
    struct valid_line_list *vll_head = NULL;

    while(getline(&lineptr, &bufsiz, cd->fp) != -1) {
        fix_trailing_nls(lineptr, strlen(lineptr));
        if(stroprnt(lineptr)) {
            goto next_ignore;
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
                        this_is_a_valid_line();
                    } else {
                        if(!has_file_ext(cached_filename, ".gz")) {
                            cached_filename_endptr[-3] = 0;
                            if(!has_file_ext(cached_filename, ".tar")) {
                                cached_filename_endptr[-3] = '.';
                                this_is_a_valid_line();
                            }
                        }
                    } 
                } else if (errno != ENOENT) {
                    log_error(stat);
                }
            }     
        }

next_ignore:
        free(lineptr);
next:
        lineptr = NULL;
    }

    free_reset_ptr(lineptr);    

    replace_file_content(cd->cachedesc_path, &cd->fp, vll_head);
    vll_cur = NULL;
    vll_head = NULL;
}

#undef add_valid_line
#undef this_is_a_valid_line

void fix_broken_cache(const struct cachedesc *cd) {
    fseek(cd->fp, 0, SEEK_SET);

    struct dirent *dent;

    rewinddir(cd->cachedir);

    errno = 0;
    while((dent = readdir(cd->cachedir))) {
        if(!strcmp(dent->d_name, ".") || !strcmp(dent->d_name, "..")) {
            goto next1;
        }

        char abspath[PATH_MAX + 1];
        memset(abspath, 0, PATH_MAX + 1);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
        snprintf(abspath, PATH_MAX, "%s/%s", cd->cachedir_path, dent->d_name);
#pragma GCC diagnostic pop

        struct stat statbuf;
        if(!stat(abspath, &statbuf) && S_ISREG(statbuf.st_mode)) {
            if(!strcmp(dent->d_name, "cachedesc")) {
                goto next1;
            }

            if(!has_file_ext(dent->d_name, ".mtx")) {
                goto next1;
            } else {
                if(!has_file_ext(dent->d_name, ".gz")) {
                    char* d_name_endptr = dent->d_name + strlen(dent->d_name);

                    d_name_endptr[-3] = 0;
                    if(!has_file_ext(dent->d_name, ".tar")) {
                        d_name_endptr[-3] = '.';
                        goto next1;
                    }
                }
            }
        } else if(errno) {
            log_error(stat);
        }

        if(remove(abspath)) {
            if(S_ISDIR(statbuf.st_mode)) {
                remove_directory_recursive(abspath);
            } else {
                log_error(remove);
            }
        }

next1:
        errno = 0;
    }

    if(errno) {
        log_error(readdir);
    }
}

void update_cachedesc_with_csum(struct cachedesc *cd, char* md5sum_stdout) {
    fseek(cd->fp, 0, SEEK_SET);

    char* new_md5csum = strtok(md5sum_stdout, " ");
    char* new_filename = strtok(NULL, " ");

    char* lineptr = NULL;
    size_t lnsz;

    long prevpos = ftell(cd->fp);

    while(getline(&lineptr, &lnsz, cd->fp) != -1) {
        char* filename = strtok(lineptr, " ");
        if(filename) {
            if(!strcmp(filename, new_filename)) {
                fseek(cd->fp, prevpos, SEEK_SET);
                goto finish;
            }
        }

        prevpos = ftell(cd->fp);
        free_reset_ptr(lineptr);
    }

finish:
    free_reset_ptr(lineptr);
    fprintf(cd->fp, "%s %s\n", new_filename, new_md5csum);
    fix_broken_cachedesc(cd);
}

int get_csum_from_cachedesc(const struct cachedesc *cd, const char* filename, char* out, size_t outsz) {
    fseek(cd->fp, 0, SEEK_SET);

    char *lineptr = NULL;
    size_t sz;
    int rv = 0;

    while(getline(&lineptr, &sz, cd->fp) != -1) {
        char* my_filename = strtok(lineptr, " ");
        char* my_md5csum = strtok(NULL, " ");

        if(!strcmp(my_filename, filename)) {
            size_t md5csumlen = strlen(my_md5csum); //MD5 hash is fixed length but who knows...
            size_t cpysz = outsz < md5csumlen ? outsz : md5csumlen;
            memcpy(out, my_md5csum, cpysz);
            goto finish1;
        }

        free_reset_ptr(lineptr);
    }

    rv = -1;

finish1:
    free_reset_ptr(lineptr);
    return rv;
}

