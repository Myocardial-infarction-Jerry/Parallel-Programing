#include <iostream>
#include <fstream>
#include <vector>

#include "utils.hpp"
#include "img2col.hpp"

int main(int argc, char const *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image> <kernel> [<kernel> [...]]" << std::endl;
        return 1;
    }

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

        if (kernel.height != kernels[0].height || kernel.width != kernels[0].width || kernel.channels != kernels[0].channels) {
            std::cerr << "Kernel " << argv[i] << " has different dimensions." << std::endl;
            return 1;
        }
    }

    Layer result = Img2Col(image, kernels, 1);

    return 0;
}
