#include<file-util/tracking/checksum.h>
#include<file-util/tracking/cachedesc.h>
#include<file-util/tracking/tracking.h>
#include<utils.h>
#include<string.h>

int main(int ac, char** argv) {
    struct cachedesc *cd;
    open_cachedir("./cachedir", &cd);

    fix_broken_cachedesc(cd);
    fix_broken_cache(cd);

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

    struct md5 out;
    if(md5sum(argv[1], &out)) {
        puts("gone wrong...");
    } else {
        printf("%s\n", out.md5);
        printf("%s\n", out.md5sum_output);
        free_reset_ptr(out.md5sum_output);   
    }

   add_file_to_track("ciao mondo");
   add_file_to_track("weowowowowo w w w");
   struct tracking_files *tf = add_file_to_track(NULL);
   free_tracking_files(&tf);

    return 0;
}