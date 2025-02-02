#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <string.h>
#include <utils.h>
#include <file-util/tracking/checksum.h>

int md5sum(const char *filename, struct md5 *out) {
    size_t filenamelen = strlen(filename);
    size_t cmdlinelen = strlen(filename) + sizeof("md5sum ") + sizeof(" 2>&1");

    char *cmdline = checked_malloc(char, cmdlinelen + 1);
    memset(cmdline, 0, cmdlinelen + 1);
    
    snprintf(cmdline, cmdlinelen, "md5sum %s 2>&1", filename);

    FILE* f = popen(cmdline, "r");
    if(f == NULL) {
        free_reset_ptr(cmdline);
        return -1;
    }

    char checksum_as_str_stdout[33];
    memset(checksum_as_str_stdout, 0, 33);
    fgets(checksum_as_str_stdout, 32, f);

    if(pclose(f) != 0) {
        free_reset_ptr(cmdline);
        return -1;
    }

    memcpy(out->md5, checksum_as_str_stdout, 33);

    size_t fake_stdoutlen = filenamelen + 2 + 32; 
    char* fake_stdout = checked_malloc(char, fake_stdoutlen + 1);
    memset(fake_stdout, 0, fake_stdoutlen + 1);
    snprintf(fake_stdout, fake_stdoutlen, "%s  %s", checksum_as_str_stdout, filename);

    out->md5sum_output = fake_stdout;

    free_reset_ptr(cmdline);
    return 0;
}