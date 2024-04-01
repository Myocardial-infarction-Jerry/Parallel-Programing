#include <vector>
#include <iostream>
#include <mpi.h>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

#define MASTER_RANK 0

struct MatrixSize {
    int m;
    int n;
    int k;
};

void matMul(float *A, float *B, float *C, int m, int n, int k) {
    // Perform matrix multiplication C = A * B
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0;
            for (int l = 0; l < n; l++)
                C[i * k + j] += A[i * n + l] * B[l * k + j];
        }
}

int main(int argc, char const *argv[]) {
    int rank, size;
    MPI_Init(&argc, (char ***)&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype matrixSizeType;
    MPI_Type_contiguous(3, MPI_INT, &matrixSizeType);
    MPI_Type_commit(&matrixSizeType);

    int M, m, n, k;
    MatrixSize matrixSize;
    float *A, *B, *C;

    if (rank == MASTER_RANK) {
        // Set the matrix dimensions
        M = n = k = 512;
        std::cerr << "Calculating for m = " << M << ", n = " << n << ", k = " << k << std::endl;
        std::cerr << "Running on " << size << " processes\n";

        // Calculate the submatrix size for each process
        m = (M + size - 1) / size;
        std::cerr << "subM = " << m << std::endl;

        // Allocate memory for matrices A, B, and C
        A = new float[m * size * n];
        B = new float[n * k];
        C = new float[m * size * k];

        // Initialize matrices A and B with random values
        for (int i = 0; i < M * n; i++) A[i] = (float)(rand()) / RAND_MAX;
        for (int i = 0; i < n * k; i++) B[i] = (float)(rand()) / RAND_MAX;

        // Set the dimensions of the matrixSize struct
        matrixSize.m = m;
        matrixSize.n = n;
        matrixSize.k = k;
    }

    auto startTime = MPI_Wtime();

    // Broadcast the matrixSize struct to all processes
    MPI_Bcast(&matrixSize, 1, matrixSizeType, MASTER_RANK, MPI_COMM_WORLD);

    // Update the local dimensions based on the received matrixSize
    m = matrixSize.m;
    n = matrixSize.n;
    k = matrixSize.k;

    // Allocate memory for the submatrices
    float *subA = new float[m * n];
    B = new float[n * k];
    float *subC = new float[m * k];

    // Scatter matrix A to all processes
    MPI_Scatter(A, m * n, MPI_FLOAT, subA, m * n, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

    // Broadcast matrix B to all processes
    MPI_Bcast(B, n * k, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

    // Perform matrix multiplication on the submatrices
    matMul(subA, B, subC, m, n, k);

    // Gather the submatrices C from all processes to the master process
    MPI_Gather(subC, m * k, MPI_FLOAT, C, m * k, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

    auto endTime = MPI_Wtime();

    if (rank == MASTER_RANK) {
        // Print the execution time
        std::cout << "matMul time: " << endTime - startTime << " seconds" << std::endl;

        // Create a JSON object to store the matrix data
        nlohmann::json output;
        output["m"] = M;
        output["n"] = n;
        output["k"] = k;
        output["A"] = nlohmann::json::array();
        output["B"] = nlohmann::json::array();
        output["C"] = nlohmann::json::array();

        // Store the values of matrices A, B, and C in the JSON object
        for (int i = 0; i < M * n; i++) output["A"].push_back(A[i]);
        for (int i = 0; i < n * k; i++) output["B"].push_back(B[i]);
        for (int i = 0; i < M * k; i++) output["C"].push_back(C[i]);

        // Write the JSON object to a file
        std::ofstream outputFile("output.json");
        outputFile << output.dump(4);
        outputFile.close();

        // Deallocate memory for matrices A and C
        delete[] A;
        delete[] C;
    }

    // Deallocate memory for submatrices
    delete[] subA;
    delete[] B;
    delete[] subC;

    // Free the MPI datatype
    MPI_Type_free(&matrixSizeType);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}