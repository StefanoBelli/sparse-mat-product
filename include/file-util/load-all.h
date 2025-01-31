#ifndef CHECK_FILES_H
#define CHECK_FILES_H

#include <stdio.h>
#include "read-config.h"

FILE** load_all_files(const char* basedir, const struct matrix_file_info** matrices);

#endif