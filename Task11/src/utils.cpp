// ------------------------------------
// Include necessary headers
// ------------------------------------
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <stb_image.h>

// ------------------------------------
// Definition of the Layer class
// ------------------------------------
Layer::Layer() {
    height = 0;
    width = 0;
    channels = 0;
    data = NULL;
}

Layer::Layer(int height, int width, int channels) {
    this->height = height;
    this->width = width;
    this->channels = channels;
    data = new float[height * width * channels];
}

Layer::~Layer() {
    delete[] data;
}

void Layer::loadImage(const char *filename) {
    // Delete previous data if any
    delete[] data;

    // Load image from file using stb_image library
    auto imageData = stbi_load(filename, &width, &height, &channels, 0);

    if (imageData == NULL) {
        std::cerr << "Error loading image: " << filename << std::endl;
        exit(1);
    }

    // Copy image data to the layer
    data = new float[height * width * channels];
    for (int i = 0; i < height * width * channels; i++)
        data[i] = imageData[i] / 255.0f;

    stbi_image_free(imageData);

    std::cout << "Loaded image: " << filename << " (" << width << "x" << height << "x" << channels << ")" << std::endl;
}

// ------------------------------------
// Definition of the Kernel class
// ------------------------------------
Kernel::Kernel() {
    height = 0;
    width = 0;
    channels = 0;
    data = NULL;
}

Kernel::Kernel(int height, int width, int channels) {
    this->height = height;
    this->width = width;
    this->channels = channels;
    data = new float[height * width * channels];
}

Kernel::~Kernel() {
    delete[] data;
}

void Kernel::loadKernel(const char *filename) {
    // Delete previous data if any
    delete[] data;

    // Load kernel from file
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening kernel file: " << filename << std::endl;
        exit(1);
    }

    // Read kernel dimensions and data from file
    file.read((char *)&height, sizeof(int));
    file.read((char *)&width, sizeof(int));
    file.read((char *)&channels, sizeof(int));
    data = new float[height * width * channels];
    file.read((char *)data, height * width * channels * sizeof(float));
    file.close();

    std::cout << "Loaded kernel: " << filename << " (" << width << "x" << height << "x" << channels << ")" << std::endl;

    // Normalize the kernel data
    float sum = 0.0f;
    for (int i = 0; i < height * width * channels; i++)
        sum += data[i];

    for (int i = 0; i < height * width * channels; i++)
        data[i] /= sum;
} 