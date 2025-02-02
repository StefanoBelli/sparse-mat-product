#include <string.h>
#include <matrices.h>
#include <utils.h>
#include <file-util/tracking/tracking.h>
#include <file-util/tracking/utils.h>
#include <file-util/tracking/cachedesc.h>

struct parsed_input_string {
    char *group;
    char *name;
};

int parse_input_string(const char* is, struct parsed_input_string* out) {
    size_t islen = strlen(is);

    char* my_is = checked_malloc(char, islen + 1);
    memset(my_is, 0, islen + 1);
    memcpy(my_is, is, islen);

    fix_trailing_nls(my_is, islen);
    if(stroprnt(my_is)) {
        free_reset_ptr(my_is);
        return -1;
    }

    char* group = strtok(my_is, " ");
    if(group == NULL) {
        free_reset_ptr(my_is);
        return -1;
    }

    char* name = strtok(NULL, " ");
    if(name == NULL) {
        free_reset_ptr(my_is);
        return -1;
    }

    size_t grouplen = strlen(group);
    size_t namelen = strlen(name);

    out->group = checked_malloc(char, grouplen + 1);
    out->name = checked_malloc(char, namelen + 1);

    memset(out->group, 0, grouplen + 1);
    memset(out->name, 0, namelen + 1);

    memcpy(out->group, group, grouplen);
    memcpy(out->name,  name, namelen);

    free_reset_ptr(my_is);
    return 0;
}

struct tracking_files *add_file_to_track(const char* s) {
    static struct tracking_files *current = NULL;

    if(s == NULL) {
        struct tracking_files *current_out = current;
        current = NULL;

        return current_out;
    }

    if(current == NULL) {
        size_t dflsize = sizeof(matrices) / sizeof(char*);
        current = checked_malloc(struct tracking_files, 1);
        current->m = checked_malloc(struct tracked_file_name, dflsize);
        current->len = 0;
        current->cap = dflsize;
    }

    struct parsed_input_string pis;
    if(parse_input_string(s, &pis)) {
        return (struct tracking_files*) -1;
    }

    if(current->len + 1 > current->cap) {
        current->cap += 10;
        current->m = checked_realloc(current->m, struct tracked_file_name, current->cap);
    }

    current->m[current->len].group_name = pis.group;
    current->m[current->len].file_name = pis.name;
    current->len++;

    return NULL;
}

void free_tracking_files(struct tracking_files **tf) {
    for(int i = 0; i < (*tf)->len; i++) {
        free_reset_ptr((*tf)->m[i].group_name);
        free_reset_ptr((*tf)->m[i].file_name);
    } 

    free_reset_ptr((*tf)->m);
    free_reset_ptr(*tf);
}

int track_files(const char* mtxfilesdir, struct tracking_files *tf) {
    struct cachedesc *cd;
    open_cachedir(mtxfilesdir, &cd);

    // make sure this ordering is right
    fix_broken_cache(cd);
    fix_broken_cachedesc(cd);

    //make outer loop iterating through tracking files
    
    struct dirent *dent;

    rewinddir(cd->cachedir);
    
    errno = 0;
    while((dent = readdir(cd->cachedir))) {
        // no "cachedesc" file from checking

        /*
        if file.mtx exists and checksum exists and checksum matches {
            pass
        }
        else if file.tar.gz exists and checksum exists and checksum matches {
            decompress, extract archive
            calculate md5sum of mtx file
            store md5sum of mtx file in cachedesc
            pass
        }
        else {
            download file.tar.gz from internet
            calculate md5sum of tar.gz file
            store md5sum of tar.gz file in cachedesc
            decompress, extract archive
            calculate md5sum of mtx file
            store md5sum of mtx file in cachedesc
            pass
        }
        */
        errno = 0;
    }

    if(errno) {
        // readdir has failed
    }

    close_cachedir(&cd);
}