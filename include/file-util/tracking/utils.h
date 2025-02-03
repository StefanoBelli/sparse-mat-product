#ifndef FILE_UTIL_UTILS_H
#define FILE_UTIL_UTILS_H

#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sys/stat.h>

#define mkcachedir(cdname) \
    (mkdir(cdname, \
        S_IRUSR | \
        S_IRGRP | \
        S_IROTH | \
        S_IWUSR | \
        S_IXUSR | \
        S_IXGRP | \
        S_IXOTH ) && errno != EEXIST)

static inline int has_file_ext(const char* filename, const char* ext) {
    char *dotat = strrchr(filename, '.');
    if(dotat == NULL) {
        return 1;
    }

    return strcmp(dotat, ext);
}

static inline int stroprnt(const char* s) {
    for(; *s; s++) {
        if(!isprint(*s)) {
            return -1;
        }
    }

    return 0;
}

static inline void fix_trailing_nls(char* s, size_t slen) {
    if(slen >= 1 && s[slen - 1] == '\n') {
        s[slen - 1] = 0;
    }

    if(slen >= 2 && s[slen - 2] == '\r') {
        s[slen - 2] = ' '; //strtok will take care
    }
}

#endif