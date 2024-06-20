#ifndef UTILS_HPP
#define UTILS_HPP

class Layer {
public:
    int height, width, channels;
    float *data;

    Layer();
    Layer(const Layer &layer);
    Layer &operator=(const Layer &layer);
    Layer(int height, int width, int channels);
    ~Layer();

    void loadImage(const char *filename);
};

class Kernel {
public:
    int height, width, channels;
    float *data;

    Kernel();
    Kernel(const Kernel &kernel);
    Kernel &operator=(const Kernel &kernel);
    Kernel(int height, int width, int channels);
    ~Kernel();

    void loadKernel(const char *filename);
};

#endif