#ifndef LOAD_ALL_FILE_UTIL_H
#define LOAD_ALL_FILE_UTIL_H

#include <stdio.h>

struct matrix_file_info {
    const char* group;
    const char* matrix;
};

struct matrix_loaded_file {
    const char* name;
    FILE* fp;
};

struct matrix_loaded_file *load_all_files(
    const char *basedir, 
    const struct matrix_file_info *additionals,
    int num_more,
    int *loaded_num);

#endif