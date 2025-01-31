#ifndef READ_CONFIG_H
#define READ_CONFIG_H

struct matrix_file_info {
    const char* group_name;
    const char* matrix_name;
};

struct matrix_file_info** read_config_file(const char* filename);
void free_matrix_file_info(struct matrix_file_info** mfi);

#endif