#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void matTranspose(float *d_A, float *d_B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        d_B[j * n + i] = d_A[i * n + j];
    }
}

void verifyTranspose(float *h_A, float *h_B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (h_A[i * n + j] != h_B[j * n + i]) {
                printf("Verification failed!\n");
                return;
            }
        }
    }
    printf("Verification passed!\n");
}

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    float *h_A = new float[n * n];
    float *h_B = new float[n * n];
    float *d_A, *d_B;

    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));

    for (int i = 0; i < n * n; i++)
        h_A[i] = rand();

    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);

    auto beginTime = std::chrono::high_resolution_clock::now();
    matTranspose << <grid, block >> > (d_A, d_B, n);
    cudaDeviceSynchronize();
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - beginTime;
    printf("Elapsed time: %lf seconds\n", elapsed.count());

    cudaMemcpy(h_B, d_B, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    verifyTranspose(h_A, h_B, n);

    delete[] h_A;
    delete[] h_B;
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
