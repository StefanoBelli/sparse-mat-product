#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>

#define checked_malloc(ty, ne) ({ \
    ty* ptr = (ty*) malloc(sizeof(ty) * (ne)); \
    if(ptr == NULL) { \
        log_error(malloc); \
        exit(EXIT_FAILURE); \
    }; ptr; \
})

#define checked_calloc(ty, ne) ({ \
    ty* ptr = (ty*) calloc((ne), sizeof(ty)); \
    if(ptr == NULL) { \
        log_error(calloc); \
        exit(EXIT_FAILURE); \
    }; ptr; \
})

#define checked_realloc(pt, ty, ne) ({ \
    ty* ptr = (ty*) realloc(pt, sizeof(ty) * (ne)); \
    if(ptr == NULL) { \
        log_error(realloc); \
        exit(EXIT_FAILURE); \
    }; ptr; \
})

#define free_reset_ptr(ptr) \
    do { \
        if((ptr)) { \
            free((ptr)); \
            (ptr) = NULL; \
        } \
    } while(0)

#define __to_s(a) #a
#define to_s(x) __to_s(x)

#define log_error(msg) \
    printf("ERROR [" __FILE__ ":" to_s(__LINE__) "] %s - " #msg ": %s\n", __func__, strerror(errno))

#define log_warn(fmt, ...) \
    printf("WARNING [" __FILE__ ":" to_s(__LINE__) "] %s - " fmt "\n", __func__, __VA_ARGS__)

#define log_warn_simple(msg) \
    printf("WARNING [" __FILE__ ":" to_s(__LINE__) "] %s - " msg "\n", __func__)

#define mkcachedir(cdname) ({ \
    errno = 0; \
    int res = (mkdir(cdname, \
        S_IRUSR | \
        S_IRGRP | \
        S_IROTH | \
        S_IWUSR | \
        S_IXUSR | \
        S_IXGRP | \
        S_IXOTH ) && errno != EEXIST); \
    res; \
})

#endif