#include <iostream>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <pthread.h>

#define NUM_THREADS 4

struct ThreadData {
    float *A, *B, *C;
    int m, n, k;
    int startRow, endRow;
};

void *matMulThread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    float *A = data->A;
    float *B = data->B;
    float *C = data->C;
    int m = data->m;
    int n = data->n;
    int k = data->k;
    int startRow = data->startRow;
    int endRow = data->endRow;

    // Perform matrix multiplication C = A * B for the assigned rows
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < k; ++j) {
            C[i * k + j] = 0;
            for (int l = 0; l < n; ++l) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }

    pthread_exit(NULL);
}

void matMul(float *A, float *B, float *C, int m, int n, int k) {
    // Create thread data and initialize thread attributes
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Calculate the number of rows to be assigned to each thread
    int rowsPerThread = m / NUM_THREADS;
    int remainingRows = m % NUM_THREADS;

    // Assign rows to each thread
    int startRow = 0;
    for (int i = 0; i < NUM_THREADS; ++i) {
        int endRow = startRow + rowsPerThread;
        if (i < remainingRows)
            endRow++;

        // Create thread data
        ThreadData *data = new ThreadData;
        data->A = A;
        data->B = B;
        data->C = C;
        data->m = m;
        data->n = n;
        data->k = k;
        data->startRow = startRow;
        data->endRow = endRow;

        // Create thread and pass thread data
        int rc = pthread_create(&threads[i], &attr, matMulThread, (void *)data);
        if (rc) {
            std::cerr << "Error: Unable to create thread, return code: " << rc << std::endl;
            exit(-1);
        }

        startRow = endRow;
    }

    // Wait for all threads to complete
    pthread_attr_destroy(&attr);
    for (int i = 0; i < NUM_THREADS; ++i) {
        int rc = pthread_join(threads[i], NULL);
        if (rc) {
            std::cerr << "Error: Unable to join thread, return code: " << rc << std::endl;
            exit(-1);
        }
    }
}

int main(int argc, char *argv[]) {
    int m, n, k;
    std::cin >> m >> n >> k;
    std::cerr << "Calculating for m = " << m << ", n = " << n << ", k = " << k << std::endl;
    std::cerr << "Running on " << NUM_THREADS << " processes\n";

    // Allocate memory for matrices A, B, and C
    float *A = new float[m * n];
    float *B = new float[n * k];
    float *C = new float[m * k];

    // Initialize matrices A and B with random values
    for (int i = 0; i < m * n; ++i) {
        A[i] = (float)(rand()) / RAND_MAX;
    }
    for (int i = 0; i < n * k; ++i) {
        B[i] = (float)(rand()) / RAND_MAX;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication C = A * B
    matMul(A, B, C, m, n, k);

    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    // Print the execution time
    std::cout << "matMul time: " << std::fixed << std::setprecision(6) << duration / 1000.0f << " seconds" << std::endl;

    // Create a JSON object to store the matrix data
    nlohmann::json output;
    output["m"] = m;
    output["n"] = n;
    output["k"] = k;
    output["A"] = nlohmann::json::array();
    output["B"] = nlohmann::json::array();
    output["C"] = nlohmann::json::array();

    // Store the values of matrices A, B, and C in the JSON object
    for (int i = 0; i < m * n; ++i) {
        output["A"].push_back(A[i]);
    }
    for (int i = 0; i < n * k; ++i) {
        output["B"].push_back(B[i]);
    }
    for (int i = 0; i < m * k; ++i) {
        output["C"].push_back(C[i]);
    }

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
    std::system("python3 src/PythonMatMul.py");

    return 0;
}