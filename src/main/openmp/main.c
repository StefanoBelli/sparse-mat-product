#include<file-util/internals/cachedesc.h>

int main() {
    struct cachedesc *cd;
    open_cachedir("./cachedir", &cd);

    printf("ptr = %p\n", cd);

    fix_broken_cachedesc(cd);
    fix_broken_cache(cd);

    puts("fix broken cache");

    char buf[100];
    memset(buf, 0, 100);
    snprintf(buf, 100, "ch1 ciao.tar.gz");

    update_cachedesc_with_csum(cd, buf);

    char buf2[100];
    memset(buf2, 0, 100);
    snprintf(buf2, 100, "ch2 another.mtx");
    update_cachedesc_with_csum(cd, buf2);

    char buf3[100];
    memset(buf3, 0, 100);

    int rv = get_csum_from_cachedesc(cd, "mm.mtx", buf3, 9);

    printf("%d = %s\n", rv, buf3);
    close_cachedir(&cd);

    return 0;
}