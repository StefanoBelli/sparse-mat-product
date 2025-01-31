#include <matrix/represent.h>

__global__ void Kernel(void) {

}

int main() {
    Kernel<<<1,1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}