#include <iostream>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <pthread.h>

#define NUM_THREADS 4

struct ThreadData {
    int *array;
    int size;
    int startIdx;
    int endIdx;
    int sum;
};

void *sumArrayThread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int *array = data->array;
    int size = data->size;
    int startIdx = data->startIdx;
    int endIdx = data->endIdx;
    int sum = 0;

    // Calculate the sum of the assigned portion of the array
    for (int i = startIdx; i < endIdx; ++i)
        sum += array[i];

    data->sum = sum;
    pthread_exit(NULL);
}

int sumArray(int *array, int size) {
    // Create thread data and initialize thread attributes
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Calculate the number of elements to be assigned to each thread
    int elementsPerThread = size / NUM_THREADS;
    int remainingElements = size % NUM_THREADS;

    // Assign elements to each thread
    int startIdx = 0;
    ThreadData threadData[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i) {
        int endIdx = startIdx + elementsPerThread;
        if (i < remainingElements)
            endIdx++;

        // Create thread data
        threadData[i].array = array;
        threadData[i].size = size;
        threadData[i].startIdx = startIdx;
        threadData[i].endIdx = endIdx;

        // Create thread and pass thread data
        int rc = pthread_create(&threads[i], &attr, sumArrayThread, (void *)&threadData[i]);
        if (rc) {
            std::cerr << "Error: Unable to create thread, return code: " << rc << std::endl;
            exit(-1);
        }

        startIdx = endIdx;
    }

    // Wait for all threads to complete
    pthread_attr_destroy(&attr);
    int totalSum = 0;
    for (int i = 0; i < NUM_THREADS; ++i) {
        int rc = pthread_join(threads[i], NULL);
        if (rc) {
            std::cerr << "Error: Unable to join thread, return code: " << rc << std::endl;
            exit(-1);
        }

        totalSum += threadData[i].sum;
    }

    return totalSum;
}

int main(int argc, char *argv[]) {
    int n;
    std::cin >> n;
    std::cerr << "Calculating sum for array of size " << n << std::endl;

    // Allocate memory for the array
    int *array = new int[n];

    // Initialize the array with random values
    for (int i = 0; i < n; ++i)
        array[i] = rand() % 3 - 1;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Calculate the sum of the array
    int sum = sumArray(array, n);

    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    // Print the execution time
    std::cerr << "vecAdd time: " << std::fixed << std::setprecision(6) << duration / 1000.0f << " seconds" << std::endl;

    // Create a JSON object to store the matrix data
    nlohmann::json output;
    output["n"] = n;
    output["array"] = nlohmann::json::array();
    output["sum"] = sum;

    // Store the values of matrices A, B, and C in the JSON object
    for (int i = 0; i < n; ++i)
        output["array"].push_back(array[i]);

    // Write the JSON object to a file
    std::ofstream outputFile("output.json");
    outputFile << output.dump(4);
    outputFile.close();

    // Deallocate memory for the array
    delete[] array;

    // Verify the result with Python script
    std::cerr << "Verifying with Python script" << std::endl;
    std::system("python3 PythonVecSum.py");

    return 0;
}