#include "img2col.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// CUDA kernel for img2col operation
__global__ void img2col(float *input, int inputHeight, int inputWidth, int inputChannels, float *output, int outputHeight, int outputWidth, int outputChannels, float *kernels, int kernelHeight, int kernelWidth, int stride) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z;

    if (i < outputHeight && j < outputWidth) {
        for (int l = 0; l < kernelHeight; ++l)
            for (int m = 0; m < kernelWidth; ++m)
                for (int n = 0; n < inputChannels; ++n)
                    output[(i * outputWidth + j) * outputChannels + k] += input[(i * stride + l) * inputWidth * inputChannels + (j * stride + m) * inputChannels + n] * kernels[k * kernelHeight * kernelWidth * inputChannels + l * kernelWidth * inputChannels + m * inputChannels + n];
    }
}

// Function to perform img2col operation
Layer Img2Col(const Layer &image, std::vector<Kernel> kernels, int stride) {
    int kernelHeight = kernels[0].height;
    int kernelWidth = kernels[0].width;

    auto kernelData = new float[kernels.size() * kernelHeight * kernelWidth * image.channels];
    for (int i = 0; i < kernels.size(); ++i)
        for (int j = 0; j < kernelHeight; ++j)
            for (int k = 0; k < kernelWidth; ++k)
                for (int l = 0; l < image.channels; ++l)
                    kernelData[i * kernelHeight * kernelWidth * image.channels + j * kernelWidth * image.channels + k * image.channels + l] = kernels[i].data[j * kernelWidth * image.channels + k * image.channels + l];

    int paddingHeight = ((image.height - 1) * stride + kernelHeight - image.height) / 2;
    int paddingWidth = ((image.width - 1) * stride + kernelWidth - image.width) / 2;

    int inputHeight = image.height + 2 * paddingHeight;
    int inputWidth = image.width + 2 * paddingWidth;
    int inputChannels = image.channels;

    auto paddingImage = new float[inputHeight * inputWidth * image.channels];
    for (int i = 0; i < image.height; ++i)
        for (int j = 0; j < image.width; ++j)
            for (int k = 0; k < image.channels; ++k)
                paddingImage[(i + paddingHeight) * inputWidth * inputChannels + (j + paddingWidth) * inputChannels + k] = image.data[i * image.width * image.channels + j * image.channels + k];

    int outputHeight = (image.height + 2 * paddingHeight - kernels[0].height) / stride + 1;
    int outputWidth = (image.width + 2 * paddingWidth - kernels[0].width) / stride + 1;
    int outputChannels = kernels.size();

    std::cerr << "[Img2Col] Input size:  " << inputHeight << "x" << inputWidth << "x" << inputChannels << "\n";
    std::cerr << "[Img2Col] Output size: " << outputHeight << "x" << outputWidth << "x" << outputChannels << "\n";
    std::cerr << "[Img2Col] Kernel size: " << kernelHeight << "x" << kernelWidth << "\n";
    std::cerr << "[Img2Col] Stride: " << stride << "\n";
    std::cerr << "[Img2Col] Memory allocating... \n";

    float *d_input, *d_output, *d_kernels;
    cudaMalloc(&d_input, inputHeight * inputWidth * inputChannels * sizeof(float));
    cudaMalloc(&d_output, outputHeight * outputWidth * outputChannels * sizeof(float));
    cudaMalloc(&d_kernels, kernels.size() * kernelHeight * kernelWidth * image.channels * sizeof(float));

    cudaMemcpy(d_input, paddingImage, inputHeight * inputWidth * inputChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, kernelData, kernels.size() * kernelHeight * kernelWidth * image.channels * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((outputWidth + block.x - 1) / block.x, (outputHeight + block.y - 1) / block.y, outputChannels);

    auto beginTime = std::chrono::high_resolution_clock::now();

    img2col << <grid, block >> > (d_input, inputHeight, inputWidth, inputChannels, d_output, outputHeight, outputWidth, outputChannels, d_kernels, kernelHeight, kernelWidth, stride);

    cudaDeviceSynchronize();

    auto endTime = std::chrono::high_resolution_clock::now();

    float *output = new float[outputHeight * outputWidth * outputChannels];
    cudaMemcpy(output, d_output, outputHeight * outputWidth * outputChannels * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernels);

    delete[] kernelData;
    delete[] paddingImage;

    Layer col(outputHeight, outputWidth, outputChannels);
    col.data = output;

    std::cerr << "[Img2Col] Time: " << 1E-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() << "ms\n";

    return col;
}
