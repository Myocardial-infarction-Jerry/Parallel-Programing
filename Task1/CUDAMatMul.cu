#include <cuda_runtime.h>
#include <iostream>

__global__ void matMul(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= size || col >= size)
        return;

    float sum = 0;
    for (int i = 0; i < size; ++i)
        sum += A[row * size + i] * B[i * size + col];

    C[row * size + col] = sum;
}

int main(int argc, char const *argv[]) {
    std::cerr << "Enter the size of the matrix:" << std::endl;
    int size; std::cin >> size;

    float *A, *B, *C;
    A = new float[size * size];
    B = new float[size * size];
    C = new float[size * size];

    for (int i = 0; i < size * size; ++i) {
        A[i] = (float)(rand()) / RAND_MAX;
        B[i] = (float)(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * size * sizeof(float));
    cudaMalloc(&d_B, size * size * sizeof(float));
    cudaMalloc(&d_C, size * size * sizeof(float));
    cudaMemcpy(d_A, A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * size * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);
    matMul << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Running time: " << milliseconds / 1000 << " seconds" << std::endl;

    return 0;
}