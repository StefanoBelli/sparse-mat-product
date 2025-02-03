#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <string.h>
#include <utils.h>
#include <file-util/tracking/extract-mtx.h>

int extract_mtx(const char* mtxname, const char* outdir) {
    size_t mtxnamelen = strlen(mtxname);
    size_t outdirlen = strlen(outdir);
    size_t cmdlinelen =
        sizeof("tar xf ") +
        outdirlen +
        sizeof("/") +
        mtxnamelen +
        sizeof(".tar.gz -C ") +
        outdirlen +
        sizeof("--strip-components=1 ") +
        mtxnamelen +
        sizeof("/") +
        mtxnamelen +
        sizeof(".mtx 2>&1");

    char *cmdline = checked_malloc(char, cmdlinelen + 1);
    memset(cmdline, 0, cmdlinelen + 1);

    snprintf(cmdline, cmdlinelen, 
        "tar xf %s/%s.tar.gz -C %s --strip-components=1 %s/%s.mtx 2>&1",
        outdir, mtxname, outdir, mtxname, mtxname);

    FILE *f = popen(cmdline, "r");
    if(f == NULL) {
        free_reset_ptr(cmdline);
        return -1;
    }

    int exitcode = pclose(f);
    free_reset_ptr(cmdline);
    return exitcode;
}