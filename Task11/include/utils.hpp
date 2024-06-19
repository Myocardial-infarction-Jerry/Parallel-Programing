#ifndef UTILS_HPP
#define UTILS_HPP

class Layer {
public:
    int height, width, channels;
    float *data;

    Layer();
    Layer(int height, int width, int channels);
    ~Layer();

    void loadImage(const char *filename);
};

class Kernel {
public:
    int height, width, channels;
    float *data;

    Kernel();
    Kernel(int height, int width, int channels);
    ~Kernel();

    void loadKernel(const char *filename);
};

#endif