#include <iostream>
#include <vector>
#include <chrono>
#include <cudnn.h>

#include "utils.hpp"

int main(int argc, char const *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image> <kernel> [<kernel> [...]]" << std::endl;
        return 1;
    }

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    Layer image;
    image.loadImage(argv[1]);

    std::vector<Kernel> kernels;
    for (int i = 2; i < argc; i++) {
        Kernel kernel;
        kernel.loadKernel(argv[i]);
        kernels.push_back(kernel);

        if (kernel.channels != image.channels) {
            std::cerr << "Kernel " << argv[i] << " has " << kernel.channels << " channels, but the image has " << image.channels << " channels." << std::endl;
            return 1;
        }
    }

    std::cerr << "[cuDNN] Running...\n";

    int kernelHeight = kernels[0].height;
    int kernelWidth = kernels[0].width;
    int numKernels = kernels.size();

    // Set up tensor descriptors for input, kernel, and output
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&kernelDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, image.channels, image.height, image.width);
    cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, numKernels, image.channels, kernelHeight, kernelWidth);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int n, c, h, w;
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, kernelDesc, &n, &c, &h, &w);

    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    float *inputData, *outputData, *kernelData;
    cudaMalloc(&inputData, sizeof(float) * image.channels * image.height * image.width);
    cudaMalloc(&outputData, sizeof(float) * n * c * h * w);
    cudaMalloc(&kernelData, sizeof(float) * numKernels * image.channels * kernelHeight * kernelWidth);

    // Copy data to device
    cudaMemcpy(inputData, image.data, sizeof(float) * image.channels * image.height * image.width, cudaMemcpyHostToDevice);
    // Assuming kernels are already filled into kernelData
    cudaMemcpy(kernelData, kernels[0].data, sizeof(float) * numKernels * image.channels * kernelHeight * kernelWidth, cudaMemcpyHostToDevice);

    auto beginTime = std::chrono::high_resolution_clock::now();

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, inputData, kernelDesc, kernelData, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, outputDesc, outputData);

    auto endTime = std::chrono::high_resolution_clock::now();

    // Copy output back to host
    float *hostOutput = new float[n * c * h * w];
    cudaMemcpy(hostOutput, outputData, sizeof(float) * n * c * h * w, cudaMemcpyDeviceToHost);

    cudaFree(inputData);
    cudaFree(outputData);
    cudaFree(kernelData);

    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(kernelDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);

    cudnnDestroy(cudnn);

    std::cerr << "[cuDNN] Time: " << 1E-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() << "ms\n";

    return 0;
}
