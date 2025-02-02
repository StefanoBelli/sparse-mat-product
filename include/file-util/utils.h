#ifndef FILE_UTIL_UTILS_H
#define FILE_UTIL_UTILS_H

#include <string.h>
#include <ctype.h>

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
    if(s[slen - 1] == '\n') {
        s[slen - 1] = 0;
    }
    
    if(s[slen - 2] == '\r') {
        s[slen - 2] = ' '; //strtok will take care
    }
}

#endif