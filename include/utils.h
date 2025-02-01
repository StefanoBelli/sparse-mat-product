#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

#define checked_malloc(ty, ne) ({ \
    ty* ptr = (ty*) malloc(sizeof(ty) * ne); \
    if(ptr == NULL) { \
        perror("malloc"); \
        exit(EXIT_FAILURE); \
    }; ptr; \
})

#define free_reset_ptr(ptr) \
    do { \
        free(ptr); \
        (ptr) = NULL; \
    } while(0)

#endif