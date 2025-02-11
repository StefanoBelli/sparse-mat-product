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

void log_resulting_vector_entries(const char* basedir, const char* mtxname, char ch, uint64_t m, double* y) {
    mkvecentdir(basedir);

    char buf[PATH_MAX];
    snprintf(buf, PATH_MAX, "%s/%s_%c.log", basedir, mtxname, ch);

    FILE *f = fopen(buf, "w");
    if(f == NULL) {
        fprintf(stderr, "unable to open vector-logging file %s\n", buf);
        return;
    }
    
    for(uint64_t i = 0; i < m; i++) {
        fprintf(f, "%lg\n", y[i]);
    }

    fclose(f);
}

#undef mkvecentdir