#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <string.h>
#include <utils.h>
#include <file-util/tracking/checksum.h>

int md5sum(const char *filename, struct md5 *out) {
    size_t cmdlinelen = 
        sizeof("md5sum ") +
        strlen(filename) + 
        sizeof(" 2>&1");

    char *cmdline = checked_malloc(char, cmdlinelen + 1);
    memset(cmdline, 0, cmdlinelen + 1);
    
    snprintf(cmdline, cmdlinelen, "md5sum %s 2>&1", filename);

    FILE* f = popen(cmdline, "r");
    if(f == NULL) {
        free_reset_ptr(cmdline);
        return -1;
    }

    memset(out->checksum, 0, MD5_CHECKSUM_LEN + 1);
    fgets(out->checksum, MD5_CHECKSUM_LEN + 1, f);

    int exitcode = pclose(f);
    free_reset_ptr(cmdline);
    return exitcode;
}

char* rebuild_md5sum_stdout(const char* filename, struct md5* md5) {
    size_t fake_stdoutlen = strlen(filename) + 2 + MD5_CHECKSUM_LEN; 
    char* fake_stdout = checked_malloc(char, fake_stdoutlen + 1);
    memset(fake_stdout, 0, fake_stdoutlen + 1);
    snprintf(fake_stdout, fake_stdoutlen, "%s  %s", md5->checksum, filename);

    return fake_stdout;
}