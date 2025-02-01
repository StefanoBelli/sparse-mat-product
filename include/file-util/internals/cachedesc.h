#ifndef CACHEDESC_FILE_UTIL_H
#define CACHEDESC_FILE_UTIL_H

#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <linux/limits.h>
#include "checksum.h"

#define OPEN_CACHEDIR_MKDIR_FAIL -1 
#define OPEN_CACHEDIR_OPENDIR_FAIL -2
#define OPEN_CACHEDIR_FOPEN_FAIL -3
#define OPEN_CACHEDIR_FREOPEN_FAIL -4

struct cachedesc {
    char cachedir_path[PATH_MAX + 1];
    char cachedesc_path[PATH_MAX + 1];
    DIR* cachedir;
    FILE* fp;
};

int open_cachedir(const char* cachedir, struct cachedesc** cd_out);
void fix_broken_cache(const struct cachedesc*);
void fix_broken_cachedesc(struct cachedesc*);
void update_cachedesc_with_csum(const struct cachedesc*, const char*);
void get_csum_from_cachedesc(const struct cachedesc*, const char*);
void close_cachedir(const struct cachedesc*);

#endif