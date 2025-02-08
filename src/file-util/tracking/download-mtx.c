#define _POSIX_C_SOURCE 200809L

#include <utils.h>
#include <file-util/tracking/download-mtx.h>

#define BASE_URL "http://sparse-files.engr.tamu.edu/MM"

int download_mtx(const char* groupname, const char* mtxname, const char* outdir) {
    size_t groupnamelen = strlen(groupname);
    size_t mtxnamelen = strlen(mtxname);
    size_t outdirlen = strlen(outdir);
    size_t cmdlinelen = 
        sizeof("curl -L -s -o ") + 
        outdirlen + 
        sizeof("/") + 
        mtxnamelen + 
        sizeof(".tar.gz ") + 
        sizeof(BASE_URL) + 
        sizeof("/") + 
        groupnamelen + 
        sizeof("/") + 
        mtxnamelen +
        sizeof(".tar.gz 2>&1");

    char* cmdline = checked_malloc(char, cmdlinelen + 1);
    snprintf(cmdline, cmdlinelen + 1, 
        "curl -L -s -o %s/%s.tar.gz " BASE_URL "/%s/%s.tar.gz 2>&1", 
        outdir, mtxname, groupname, mtxname);

    FILE *f = popen(cmdline, "r");
    if(f == NULL) {
        free_reset_ptr(cmdline);
        return -1;
    }

    int exitcode = pclose(f);
    free_reset_ptr(cmdline);
    return exitcode;
}