#include <iostream>
#include "parallel_for.h"

struct functor_args {
    float *A, *B, *C;
};

void *functor(int idx, void *args) {
    auto args_data = (functor_args *)args;
    args_data->C[idx] = args_data->A[idx] + args_data->B[idx];
}

int main(int argc, char const *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <num_threads>" << std::endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    float *A = new float[n];
    float *B = new float[n];
    float *C = new float[n];

    std::cerr << "Generating random numbers..." << std::endl;
    for (int i = 0; i < n; ++i) {
        A[i] = rand() * 1.0 / RAND_MAX;
        B[i] = rand() * 1.0 / RAND_MAX;
    }

    std::cerr << "Calculating vectors with " << num_threads << " threads..." << std::endl;
    functor_args args = { A, B, C };
    parallel_for(0, n, 1, functor, (void *)&args, num_threads);

    return 0;
}
