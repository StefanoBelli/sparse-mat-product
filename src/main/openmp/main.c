#include <executor.h>

double kernel_csr(const void *format, const union format_args *format_args) {
    return 0;
}

double kernel_hll(const void *format, const union format_args *format_args) {
    return 0;
}

int main(int argc, char** argv) {
    struct kernel_execution_info kexi[2] = {
        {
            .kernel = kernel_csr,
            .format = CSR,
        },
        {
            .kernel = kernel_hll,
            .format = HLL,
            .hll_hack_size = 32,
        }
    };

    struct executor_args eargs = {
        .nkexs = 2,
        .runner = CPU_SERIAL,
        .kexinfos = kexi
    };

    run_executor(argc, argv, &eargs);
    return 0;
}