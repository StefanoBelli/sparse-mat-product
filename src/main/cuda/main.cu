#include <cuda/helper_cuda.h>

extern "C" {
#include <config.h>
}

__global__ void Kernel(void) {

}

int main(int argc, char** argv) {
    //setup(argc, argv);

    Kernel<<<1,1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}