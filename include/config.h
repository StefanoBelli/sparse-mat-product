#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>

struct opened_mtx_file_list {
    FILE *fp;
    char *name;
    struct opened_mtx_file_list *next;
};

struct opened_mtx_file_list* setup(int argc, char** argv);
void free_all_opened_mtxs(struct opened_mtx_file_list**);

#endif