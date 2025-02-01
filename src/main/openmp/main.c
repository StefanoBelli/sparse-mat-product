#include<file-util/internals/cachedesc.h>

int main() {
    struct cachedesc *cd;
    open_cachedir("./cachedir", &cd);

    printf("ptr = %p\n", cd);

    fix_broken_cachedesc(cd);
    fix_broken_cache(cd);

    puts("fix broken cache");

    close_cachedir(&cd);

    return 0;
}