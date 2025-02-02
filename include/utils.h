#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#define checked_malloc(ty, ne) ({ \
    ty* ptr = (ty*) malloc(sizeof(ty) * ne); \
    if(ptr == NULL) { \
        log_error(malloc); \
        exit(EXIT_FAILURE); \
    }; ptr; \
})

#define checked_realloc(pt, ty, ne) ({ \
    ty* ptr = (ty*) realloc(pt, sizeof(ty) * ne); \
    if(ptr == NULL) { \
        log_error(realloc); \
        exit(EXIT_FAILURE); \
    }; ptr; \
})

#define free_reset_ptr(ptr) \
    do { \
        free(ptr); \
        (ptr) = NULL; \
    } while(0)

#define __to_s(a) #a
#define to_s(x) __to_s(x)

#define log_error(msg) \
    printf("ERROR [" __FILE__ ":" to_s(__LINE__) "] %s - " #msg ": %s\n", __func__, strerror(errno))

#endif