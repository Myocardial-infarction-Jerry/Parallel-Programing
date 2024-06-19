#ifndef IMG2COL_CUH
#define IMG2COL_CUH

#include "utils.hpp"

#include <vector>

Layer Img2Col(const Layer &image, std::vector<Kernel> kernels, int stride);

#endif