#include<file-util/internals/cachedesc.h>

int main() {
    struct cachedesc *cd;
    int res = open_cachedir("./cachedir", &cd);

    printf("res = %d, ptr = %p\n", res, cd);

    fix_broken_cachedesc(cd);
    fix_broken_cache(cd);

    puts("fix broken cache");

    close_cachedir(&cd);

    return 0;
}