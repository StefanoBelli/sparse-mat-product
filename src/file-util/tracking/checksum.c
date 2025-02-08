#define _POSIX_C_SOURCE 200809L

#include <utils.h>
#include <file-util/tracking/checksum.h>
#include <file-util/tracking/utils.h>

int md5sum(const char *filename, struct md5 *out) {
    size_t cmdlinelen = 
        sizeof("md5sum ") +
        strlen(filename) + 
        sizeof(" 2>&1");

    char *cmdline = checked_malloc(char, cmdlinelen + 1);
    snprintf(cmdline, cmdlinelen + 1, "md5sum %s 2>&1", filename);

    FILE* f = popen(cmdline, "r");
    if(f == NULL) {
        free_reset_ptr(cmdline);
        return -1;
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    fgets(out->checksum, MD5_CHECKSUM_LEN + 1, f);
#pragma GCC diagnostic pop

    fix_trailing_nls(out->checksum, strlen(out->checksum));

    int exitcode = pclose(f);
    free_reset_ptr(cmdline);
    return exitcode;
}

char* rebuild_md5sum_stdout(const char* filename, const struct md5* md5) {
    size_t fake_stdoutlen = strlen(filename) + 1 + MD5_CHECKSUM_LEN; 
    char* fake_stdout = checked_malloc(char, fake_stdoutlen + 1);
    snprintf(fake_stdout, fake_stdoutlen + 1, "%s %s", md5->checksum, filename);

    return fake_stdout;
}