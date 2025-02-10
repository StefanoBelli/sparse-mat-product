#ifndef CACHEDESC_FILE_UTIL_H
#define CACHEDESC_FILE_UTIL_H

#include <stdio.h>
#include <dirent.h>
#include <linux/limits.h>
#include "checksum.h"

struct cachedesc {
    char cachedir_path[PATH_MAX + 1];
    char cachedesc_path[PATH_MAX + 1];
    DIR* cachedir;
    FILE* fp;
};

void open_cachedir(const char* cachedir, struct cachedesc** cd_out);

#ifdef FIX_BROKEN_CACHE
void fix_broken_cache(const struct cachedesc*);
#endif

void fix_broken_cachedesc(struct cachedesc*);
void update_cachedesc_with_csum(struct cachedesc*, char*);
int get_csum_from_cachedesc(const struct cachedesc*, const char*, char**);
void close_cachedir(struct cachedesc**);

#endif