#include <utils.h>

double *make_vector_of_doubles(uint64_t nelems) {
    double *vek = checked_malloc(double, nelems);

    for(uint64_t i = 0; i < nelems; i++) {
        vek[i] = i % MAX_ELEM_VAL;
    }

    return vek;
}