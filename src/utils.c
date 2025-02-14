#include <utils.h>
#include <linux/limits.h>

double *make_vector_of_doubles(uint64_t nelems) {
    double *vek = checked_malloc(double, nelems);

    for(uint64_t i = 0; i < nelems; i++) {
        vek[i] = i % MAX_ELEM_VAL;
    }

    return vek;
}

#define mkvecentdir(name) mkcachedir(name)

static void log_resulting_vector_entries(const char* basedir, const char* mtxname, const char* mtxformat, uint64_t m, double* y) {
    mkvecentdir(basedir);

    char buf[PATH_MAX];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
    snprintf(buf, PATH_MAX, "%s/%s_%s.csv", basedir, mtxname, mtxformat);
#pragma GCC diagnostic pop

    FILE *f = fopen(buf, "w");
    if(f == NULL) {
        fprintf(stderr, "unable to open vector-logging file %s\n", buf);
        return;
    }

    fprintf(f, "y\n");

    for(uint64_t i = 0; i < m; i++) {
        fprintf(f, "%lg\n", y[i]);
    }

    fclose(f);
}

#undef mkvecentdir

void write_y_vector_to_csv(const char* runner, const char* mtxname, const char* mtxformat, uint64_t m, double* y) {
    static char current_mtx[PATH_MAX] = { 0 };
    static char current_format[PATH_MAX] = { 0 };

    if(strcmp(current_mtx, mtxname) || strcmp(current_format, mtxformat)) {
        printf(" \t\t>>> y logger: matrix or its format has changed: %s %s\n", mtxname, mtxformat);

        char pathbuf[PATH_MAX];
        snprintf(pathbuf, PATH_MAX, "yvector-%s", runner);

        log_resulting_vector_entries(pathbuf, mtxname, mtxformat, m, y);

        snprintf(current_mtx, PATH_MAX, "%s", mtxname);
        snprintf(current_format, PATH_MAX, "%s", mtxformat);
    }
}