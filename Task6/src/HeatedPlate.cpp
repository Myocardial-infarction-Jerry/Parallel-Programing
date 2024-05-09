#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <thread>
#include "parallel_for.h"

#define M 500
#define N 500

struct LoopArgs {
    double mean;
    double(*w)[N];
    double(*u)[N];
    double diff;
    double epsilon;
};

void *initialize_boundary_w(int index, void *args) {
    LoopArgs *la = static_cast<LoopArgs *>(args);
    if (index == 0) {
        for (int i = 1; i < M - 1; ++i) {
            la->w[i][0] = 100.0;
        }
    }
    else if (index == 1) {
        for (int i = 1; i < M - 1; ++i) {
            la->w[i][N - 1] = 100.0;
        }
    }
    else if (index == 2) {
        for (int j = 0; j < N; ++j) {
            la->w[M - 1][j] = 100.0;
        }
    }
    else if (index == 3) {
        for (int j = 0; j < N; ++j) {
            la->w[0][j] = 0.0;
        }
    }
    return nullptr;
}

void *compute_mean(int index, void *args) {
    LoopArgs *la = static_cast<LoopArgs *>(args);
    if (index == 0) {
        for (int i = 1; i < M - 1; ++i) {
            la->mean += la->w[i][0] + la->w[i][N - 1];
        }
    }
    else {
        for (int j = 0; j < N; ++j) {
            la->mean += la->w[M - 1][j] + la->w[0][j];
        }
    }
    return nullptr;
}

void *initialize_interior_w(int index, void *args) {
    LoopArgs *la = static_cast<LoopArgs *>(args);
    for (int j = 1; j < N - 1; ++j) {
        la->w[index][j] = la->mean;
    }
    return nullptr;
}

void *copy_u_from_w(int index, void *args) {
    LoopArgs *la = static_cast<LoopArgs *>(args);
    for (int j = 0; j < N; ++j) {
        la->u[index][j] = la->w[index][j];
    }
    return nullptr;
}

void *compute_w_from_u(int index, void *args) {
    LoopArgs *la = static_cast<LoopArgs *>(args);
    for (int j = 1; j < N - 1; ++j) {
        la->w[index][j] = (la->u[index - 1][j] + la->u[index + 1][j] + la->u[index][j - 1] + la->u[index][j + 1]) / 4.0;
    }
    return nullptr;
}

void *calculate_diff(int index, void *args) {
    LoopArgs *la = static_cast<LoopArgs *>(args);
    double local_diff = 0.0;
    for (int j = 1; j < N - 1; ++j) {
        double delta = fabs(la->w[index][j] - la->u[index][j]);
        if (local_diff < delta) {
            local_diff = delta;
        }
    }

    if (la->diff < local_diff) {
        la->diff = local_diff;
    }

    return nullptr;
}

int main(int argc, char *argv[]) {
    double diff;
    double epsilon = 0.001;
    int iterations;
    int iterations_print;
    double mean;
    double u[M][N];
    double w[M][N];
    double wtime;

    printf("\n");
    printf("HEATED_PLATE_PARALLEL\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    int num_threads = std::thread::hardware_concurrency();
    printf("  Number of processors available = %d\n", num_threads);
    printf("  Number of threads =              %d\n", num_threads);

    LoopArgs args;
    args.mean = 0.0;
    args.w = w;
    args.u = u;
    args.epsilon = epsilon;

    // Initialize boundary values
    parallel_for(0, 4, 1, initialize_boundary_w, &args, num_threads);

    // Compute mean
    parallel_for(0, 2, 1, compute_mean, &args, num_threads);

    // Normalize mean
    args.mean = args.mean / (double)(2 * M + 2 * N - 4);
    printf("\n");
    printf("  MEAN = %f\n", args.mean);

    // Initialize interior solution to mean value
    parallel_for(1, M - 1, 1, initialize_interior_w, &args, num_threads);

    // Iterate until the new solution W differs from U by no more than EPSILON
    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");
    auto beginStamp = std::chrono::high_resolution_clock::now();

    diff = epsilon;

    while (epsilon <= diff) {
        args.diff = 0.0;

        // Copy U from W
        parallel_for(0, M, 1, copy_u_from_w, &args, num_threads);

        // Compute W from U
        parallel_for(1, M - 1, 1, compute_w_from_u, &args, num_threads);

        // Calculate the difference
        parallel_for(1, M - 1, 1, calculate_diff, &args, num_threads);

        diff = args.diff;
        iterations++;
        if (iterations == iterations_print) {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }

    auto endStamp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = endStamp - beginStamp;

    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", elapsed.count());
    /*
      Terminate.
    */
    printf("\n");
    printf("HEATED_PLATE_PARALLEL:\n");
    printf("  Normal end of execution.\n");

    return 0;
}
