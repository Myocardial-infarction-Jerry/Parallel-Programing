#include <iostream>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <omp.h>

#define NUM_THREADS 16

void matMul(float *A, float *B, float *C, int m, int n, int k) {
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < n; ++l) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <k>" << std::endl;
        return 1;
    }

    // Get the values of m, n, and k from the command line
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    std::cerr << "Calculating for m = " << m << ", n = " << n << ", k = " << k << std::endl;

    // Set the number of threads

    std::cerr << "Running on " << NUM_THREADS << " threads\n";

    // Allocate memory for matrices A, B, and C
    float *A = new float[m * n];
    float *B = new float[n * k];
    float *C = new float[m * k];

    // Initialize matrices A and B with random values
    for (int i = 0; i < m * n; ++i) A[i] = (float)(rand()) / RAND_MAX;
    for (int i = 0; i < n * k; ++i) B[i] = (float)(rand()) / RAND_MAX;


    auto startTime = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication C = A * B
    matMul(A, B, C, m, n, k);
    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    // Print the execution time
    std::cout << "matMul time: " << std::fixed << std::setprecision(6) << duration / 1000000.0f << " seconds" << std::endl;

    // Create a JSON object to store the matrix data
    nlohmann::json output;
    output["m"] = m;
    output["n"] = n;
    output["k"] = k;
    output["A"] = nlohmann::json::array();
    output["B"] = nlohmann::json::array();
    output["C"] = nlohmann::json::array();

    // Store the values of matrices A, B, and C in the JSON object
    for (int i = 0; i < m * n; ++i) output["A"].push_back(A[i]);
    for (int i = 0; i < n * k; ++i) output["B"].push_back(B[i]);
    for (int i = 0; i < m * k; ++i) output["C"].push_back(C[i]);

    // Write the JSON object to a file
    std::ofstream outputFile("output.json");
    outputFile << output.dump(4);
    outputFile.close();

    // Deallocate memory for matrices A, B, and C
    delete[] A;
    delete[] B;
    delete[] C;

    // Verify the result with Python script
    std::cerr << "Verifying with Python script" << std::endl;
    std::system("python3 PythonMatMul.py");

    return 0;
}