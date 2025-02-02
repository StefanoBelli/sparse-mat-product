#ifndef CHECKSUM_FILE_UTIL_H
#define CHECKSUM_FILE_UTIL_H

struct md5 {
    char* md5sum_output;
    char md5[33];
};

int md5sum(const char* filename, struct md5* out);

#endif