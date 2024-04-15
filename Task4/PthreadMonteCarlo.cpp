#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip> 

#define NUM_THREADS 8

struct ThreadData {
    int numPoints;
    int countInCircle;
};

void *monteCarlo(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    data->countInCircle = 0;

    unsigned int randState = time(NULL);
    for (int i = 0; i < data->numPoints; ++i) {
        double x = (double)rand_r(&randState) / RAND_MAX * 2 - 1;
        double y = (double)rand_r(&randState) / RAND_MAX * 2 - 1;
        if (x * x + y * y <= 1) {
            ++data->countInCircle;
        }
    }

    return nullptr;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <numPoints>" << std::endl;
        return 1;
    }

    int numPoints = atoi(argv[1]);
    int pointsPerThread = numPoints / NUM_THREADS;
    std::cout << "Running with " << NUM_THREADS << " threads" << std::endl;

    auto start = clock();

    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; ++i) {
        threadData[i].numPoints = pointsPerThread;
        pthread_create(&threads[i], nullptr, monteCarlo, &threadData[i]);
    }

    int totalInCircle = 0;
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], nullptr);
        totalInCircle += threadData[i].countInCircle;
    }

    auto end = clock();
    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;

    double piEstimate = 4 * (double)totalInCircle / numPoints;
    std::cout << "Total points in circle: " << totalInCircle << std::endl;
    std::cout << "Pi estimate: " << std::setprecision(8) << std::fixed << piEstimate << std::endl;
    std::cout << "Running time: " << elapsedTime << " seconds" << std::endl;

    return 0;
}