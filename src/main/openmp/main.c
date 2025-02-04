#include<config.h>

int main(int argc, char** argv) {
    struct opened_mtx_file_list *head = setup(argc, argv);
    free_all_opened_mtxs(&head);
}