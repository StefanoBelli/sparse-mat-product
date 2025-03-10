#ifndef FILE_UTILS_TRACKING_H
#define FILE_UTILS_TRACKING_H

#include<stddef.h>

struct tracked_file_name {
    char* group_name;
    char* file_name;
};

struct tracking_files {
    int len;
    int cap;
    struct tracked_file_name *m;
};

struct tracking_files *add_file_to_track(const char*, size_t);
void free_tracking_files(struct tracking_files**);
void track_files(const char*, struct tracking_files*);

#endif