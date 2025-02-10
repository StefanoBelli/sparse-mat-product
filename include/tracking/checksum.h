#ifndef CHECKSUM_FILE_UTIL_H
#define CHECKSUM_FILE_UTIL_H

#define MD5_CHECKSUM_LEN 32

struct md5 {
    char checksum[MD5_CHECKSUM_LEN + 1];
};

int md5sum(const char* filename, struct md5* out);
char* rebuild_md5sum_stdout(const char* filename, const struct md5* md5);

#endif