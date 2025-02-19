#include <executor.h>
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

static void log_resulting_vector_entries(const char* basedir, const char* mtxname, uint64_t m, double* y) {
    mkvecentdir(basedir);

    char buf[PATH_MAX];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
    snprintf(buf, PATH_MAX, "%s/%s", basedir, mtxname);
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

void write_y_vector_to_csv(
        const char* runner,
        const char* variant,
        mult_datatype mdt,
        const char* mtxname, 
        const char* mtxformat, 
        uint64_t m, 
        double* y) {

    static char current_mtx[PATH_MAX] = { 0 };
    static char current_format[PATH_MAX] = { 0 };
    static const char* current_variant = 0;
    static mult_datatype current_mdt = FLOAT64;

    if(
            strcmp(current_mtx, mtxname) || 
            strcmp(current_format, mtxformat) ||
            current_variant != variant || // careful here, this works bc variant is a string in a "rodata" section
            current_mdt != mdt) {

        printf(" \t\t>>> y logger: matrix, its format, variant or datatype has changed: %s %s %s %s\n", 
            mtxname, 
            mtxformat, 
            variant == NULL ? "v1" : variant, 
            mdt == FLOAT64 ? "fp64" : "fp32");

        char pathbuf[PATH_MAX];
        snprintf(pathbuf, PATH_MAX, "yvector-%s", runner);

        char filename[PATH_MAX];
        snprintf(filename, PATH_MAX, "%s_%s_%s_%s.csv", 
            mtxname, 
            mtxformat, 
            variant == NULL ? "v1" : variant, 
            mdt == FLOAT64 ? "fp64" : "fp32");

        log_resulting_vector_entries(pathbuf, filename, m, y);

        snprintf(current_mtx, PATH_MAX, "%s", mtxname);
        snprintf(current_format, PATH_MAX, "%s", mtxformat);
        current_variant = variant;
        current_mdt = mdt;
    }
}

int ceiling_div(int m, int n) {
    return (m + n - 1) / n;
}