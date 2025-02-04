#include <string.h>
#include <math.h>
#include <utils.h>
#include <file-util/tracking/tracking.h>
#include <file-util/tracking/utils.h>
#include <file-util/tracking/cachedesc.h>
#include <file-util/tracking/extract-mtx.h>
#include <file-util/tracking/download-mtx.h>

struct parsed_input_string {
    char *group;
    char *name;
};

int parse_input_string(const char* is, struct parsed_input_string* out) {
    size_t islen = strlen(is);

    char* my_is = checked_malloc(char, islen + 1);
    memset(my_is, 0, islen + 1);
    memcpy(my_is, is, islen);

    fix_trailing_nls(my_is, islen);
    if(stroprnt(my_is)) {
        free_reset_ptr(my_is);
        return -1;
    }

    char* group = strtok(my_is, " ");
    if(group == NULL) {
        free_reset_ptr(my_is);
        return -1;
    }

    char* name = strtok(NULL, " ");
    if(name == NULL) {
        free_reset_ptr(my_is);
        return -1;
    }

    size_t grouplen = strlen(group);
    size_t namelen = strlen(name);

    out->group = checked_malloc(char, grouplen + 1);
    out->name = checked_malloc(char, namelen + 1);

    memset(out->group, 0, grouplen + 1);
    memset(out->name, 0, namelen + 1);

    memcpy(out->group, group, grouplen);
    memcpy(out->name,  name, namelen);

    free_reset_ptr(my_is);
    return 0;
}

struct tracking_files *add_file_to_track(const char* s, size_t initial_size) {
    static struct tracking_files *current = NULL;

    if(s == NULL) {
        struct tracking_files *current_out = current;
        current = NULL;

        return current_out;
    }

    if(current == NULL) {
        current = checked_malloc(struct tracking_files, 1);
        current->m = checked_malloc(struct tracked_file_name, initial_size);
        current->len = 0;
        current->cap = initial_size;
    }

    struct parsed_input_string pis;
    if(parse_input_string(s, &pis)) {
        log_warn_simple("unable to parse input string");
        return (struct tracking_files*) -1;
    }

    if(current->len + 1 > current->cap) {
        current->cap += 10;
        current->m = checked_realloc(current->m, struct tracked_file_name, current->cap);
    }

    current->m[current->len].group_name = pis.group;
    current->m[current->len].file_name = pis.name;
    current->len++;

    return NULL;
}

void free_tracking_files(struct tracking_files **tf) {
    for(int i = 0; i < (*tf)->len; i++) {
        free_reset_ptr((*tf)->m[i].group_name);
        free_reset_ptr((*tf)->m[i].file_name);
    } 

    free_reset_ptr((*tf)->m);
    free_reset_ptr(*tf);
}

static int file_exists(const char* path) {
    errno = 0;
    struct stat statbuf;
    return !stat(path, &statbuf) && errno != ENOENT && S_ISREG(statbuf.st_mode);
}

static int valid_file(struct cachedesc *cd, const char* filepath, const char* filename) {
    if(file_exists(filepath)) {
        char *checksum;
        if(get_csum_from_cachedesc(cd, filename, &checksum)) {
            return -1;
        }

        struct md5 calculated_md5;
        if(md5sum(filepath, &calculated_md5)) {
            free_reset_ptr(checksum);
            return -1;
        }

        int res = strcmp(checksum, calculated_md5.checksum);
        free_reset_ptr(checksum);
        return res;
    }

    return -1;
}

static void initialize_filename_bufs(
        char* file_buf, 
        char* filepath_buf, 
        const char* mtxname, 
        const char* cachedirpath, 
        const char* ext) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
    snprintf(file_buf, PATH_MAX + 1, "%s.%s", mtxname, ext);
    snprintf(filepath_buf, PATH_MAX + 1, "%s/%s.%s", cachedirpath, mtxname, ext);
#pragma GCC diagnostic pop
}

static int calc_and_store_md5(struct cachedesc* cd, const char *filepath, const char *filename, const char* mtxname) {
    struct md5 calculated_md5;
    if (md5sum(filepath, &calculated_md5)) {
        log_warn("md5sum failed for %s", mtxname);
        return -1;
    }

    char *md5stdout = rebuild_md5sum_stdout(filename, &calculated_md5);
    update_cachedesc_with_csum(cd, md5stdout);
    free_reset_ptr(md5stdout);

    return 0;
}

struct do_extract_args {
    const char* mtxname;
    const char* filepath_mtx;
    const char* file_mtx;
    const char* cdpath;
    struct cachedesc* cd;
};

static void do_extract(const struct do_extract_args *args) {
    if (extract_mtx(args->mtxname, args->cdpath)) {
        log_warn("tar failed for %s", args->mtxname);
    } else {
        calc_and_store_md5(args->cd, args->filepath_mtx, args->file_mtx, args->mtxname);
    }
}

struct do_download_args {
    const char* mtxname;
    const char* groupname;
    const char* filepath_targz;
    const char* filepath_mtx;
    const char* file_targz;
    const char* file_mtx;
    const char* cdpath;
    struct cachedesc* cd;
};

static void do_download_and_extract(const struct do_download_args *args) {
    if (download_mtx(args->groupname, args->mtxname, args->cdpath)) {
        log_warn("curl failed for %s", args->mtxname);
    } else {
        if (!calc_and_store_md5(args->cd, args->filepath_targz, args->file_targz, args->mtxname)) {
            struct do_extract_args a;
            a.mtxname = args->mtxname;
            a.filepath_mtx = args->filepath_mtx;
            a.file_mtx = args->file_mtx;
            a.cdpath = args->cdpath;
            a.cd = args->cd;
            do_extract(&a);
        }
    }
}

#define INIT_DO_EXTRACT_ARGS(_name) \
    struct do_extract_args _name; \
    _name.mtxname = mtxname; \
    _name.filepath_mtx = filepath_mtx; \
    _name.file_mtx = file_mtx; \
    _name.cdpath = cdpath; \
    _name.cd = cd

#define INIT_DO_DOWNLOAD_ARGS(_name) \
    struct do_download_args _name; \
    _name.mtxname = mtxname; \
    _name.filepath_mtx = filepath_mtx; \
    _name.file_mtx = file_mtx; \
    _name.cdpath = cdpath; \
    _name.cd = cd; \
    _name.groupname = groupname; \
    _name.filepath_targz = filepath_targz; \
    _name.file_targz = file_targz

static void __clear_remaining_chrs_stdout(int count) {
    for (int j = 0; j < count; j++){
        putc(' ', stdout);
    }
}

static void __conditional_last_putc(int print_newline) {
    if(print_newline) {
        puts("");
    } else {
        putc('\r', stdout);
        fflush(stdout);
    }
}

#define log_progress_now(action, status) \
    { \
        int curnumprnt; \
        printf(action " mtx file from %s: %s (%d/%d) [" status "] %n", \
            cdpath, mtxname, i + 1, tf->len, &curnumprnt); \
        __clear_remaining_chrs_stdout(abs(numprnt - curnumprnt)); \
        numprnt = curnumprnt; \
        __conditional_last_putc(i == tf->len - 1); \
    }

void track_files(const char* mtxfilesdir, struct tracking_files *tf) {
    struct cachedesc *cd;
    open_cachedir(mtxfilesdir, &cd);

    const char* cdpath = cd->cachedir_path;

    fix_broken_cachedesc(cd);
    fix_broken_cache(cd);

    int numprnt = 0;

    for (int i = 0; i < tf->len; i++) {
        const char* mtxname = tf->m[i].file_name;
        const char* groupname = tf->m[i].group_name;

        char file_mtx[PATH_MAX + 1];
        char filepath_mtx[PATH_MAX + 1];

        initialize_filename_bufs(
            file_mtx, filepath_mtx, mtxname, cdpath, "mtx");

        if(valid_file(cd, filepath_mtx, file_mtx)) {
            char file_targz[PATH_MAX + 1];
            char filepath_targz[PATH_MAX + 1];
            
            initialize_filename_bufs(
                file_targz, filepath_targz, mtxname, cdpath, "tar.gz");

            if(valid_file(cd, filepath_targz, file_targz)) {
                log_progress_now("corrupt", "download-and-extract...");
                INIT_DO_DOWNLOAD_ARGS(a);
                do_download_and_extract(&a); 
            } else {
                log_progress_now("corrupt", "extract-only...");
                INIT_DO_EXTRACT_ARGS(a);
                do_extract(&a);
            }
        } else {
            log_progress_now("good", "ok"); 
        }
    }

    close_cachedir(&cd);
}

#undef log_progress_now
#undef INIT_DO_EXTRACT_ARGS
#undef INIT_DO_DOWNLOAD_ARGS