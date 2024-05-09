#include <iostream>
#include <fstream>
#include <chrono>
#include <nlohmann/json.hpp>
#include "parallel_for.h"

struct functor_args {
    float *A, *B, *C;
    int n;
};

void *functor(int idx, void *args) {
    auto args_data = (functor_args *)args;

    int i = idx;
    int n = args_data->n;
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) {
            args_data->C[i * n + j] += args_data->A[i * n + k] * args_data->B[k * n + j];
            // printf("%d %d %d\n", i, j, k);
        }
}

int main(int argc, char const *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <num_threads>" << std::endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    float *A = new float[n * n];
    float *B = new float[n * n];
    float *C = new float[n * n];

    std::cerr << "Generating random numbers..." << std::endl;
    for (int i = 0; i < n * n; ++i) {
        A[i] = rand() * 1.0 / RAND_MAX;
        B[i] = rand() * 1.0 / RAND_MAX;
        C[i] = 0;
    }

    std::cerr << "Calculating vectors with " << num_threads << " threads..." << std::endl;
    auto beginStamp = std::chrono::high_resolution_clock::now();
    functor_args args = { A, B, C, n };
    parallel_for(0, n, 1, functor, (void *)&args, num_threads);
    auto endStamp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = endStamp - beginStamp;
    std::cerr << "Time elapsed: " << elapsed.count() << " ms" << std::endl;

    std::cerr << "Output results to file..." << std::endl;
    nlohmann::json output;

    output["n"] = n;
    output["A"] = nlohmann::json::array();
    output["B"] = nlohmann::json::array();
    output["C"] = nlohmann::json::array();
    for (int i = 0; i < n * n; ++i) {
        output["A"].push_back(A[i]);
        output["B"].push_back(B[i]);
        output["C"].push_back(C[i]);
    }

    std::ofstream outputFile("output.json");
    outputFile << output.dump(4);
    outputFile.close();

    delete[] A;
    delete[] B;
    delete[] C;

    // Run Python script to verify the results
    std::cerr << "Running Python script to verify the results..." << std::endl;
    std::system("python3 src/PythonMatMul.py");

    return 0;
}
