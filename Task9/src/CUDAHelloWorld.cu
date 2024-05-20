#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel function to print Hello World from GPU
__global__ void helloFromGPU(int n, int m, int k) {
    int idx = blockIdx.x;
    int idy = threadIdx.y + threadIdx.x * blockDim.y;

    if (idx < n && idy < m * k) {
        printf("Hello World from Thread (%d, %d) in Block %d!\n", threadIdx.x, threadIdx.y, blockIdx.x);
    }
}

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <n> <m> <k>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int k = atoi(argv[3]);

    // Define the dimension of the thread block and grid
    dim3 block(m, k);
    dim3 grid(n);

    // Launch the CUDA Kernel
    helloFromGPU << <grid, block >> > (n, m, k);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    printf("Hello World from the host!\n");

    return 0;
}