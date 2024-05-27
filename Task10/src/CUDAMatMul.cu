#include <cuda_runtime.h>
#include <iostream>

#define DIM 32

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

__global__ void matMulShared(float *A, float *B, float *C, int size) {
    __shared__ float s_A[DIM][DIM];
    __shared__ float s_B[DIM][DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    for (int i = 0; i < size / DIM; ++i) {
        s_A[threadIdx.y][threadIdx.x] = A[row * size + i * DIM + threadIdx.x];
        s_B[threadIdx.y][threadIdx.x] = B[(i * DIM + threadIdx.y) * size + col];
        __syncthreads();

        for (int j = 0; j < DIM; ++j)
            sum += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
        __syncthreads();
    }

    C[row * size + col] = sum;
}

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    int size = atoi(argv[1]);

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

    dim3 threadsPerBlock(DIM, DIM);
    dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);

    cudaEventRecord(start);
    matMul << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float counterGlobal = 0;
    cudaEventElapsedTime(&counterGlobal, start, stop);

    cudaMemcpy(C, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(start);
    matMulShared << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float counterShared = 0;
    cudaEventElapsedTime(&counterShared, start, stop);

    cudaMemcpy(C, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Global Memory: " << counterGlobal << " ms" << std::endl;
    std::cout << "Shared Memory: " << counterShared << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}