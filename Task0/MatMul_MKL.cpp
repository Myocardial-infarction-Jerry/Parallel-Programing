#include <iostream>
#include <vector>
#include <chrono>
#include <mkl.h> 

#define AUTO_GENERATE_TEST_CASES 1

std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B) {
    // Get the dimensions of the matrices
    int m = A.size();
    int n = A[0].size();
    int k = B[0].size();

    std::vector<std::vector<float>> C(m, std::vector<float>(k, 0));

    float *a = (float *)mkl_malloc(m * n * sizeof(float), 64);
    float *b = (float *)mkl_malloc(n * k * sizeof(float), 64);
    float *c = (float *)mkl_malloc(m * k * sizeof(float), 64);

    // Copy the matrices to the arrays
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            a[i * n + j] = A[i][j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            b[i * k + j] = B[i][j];

    // Perform matrix multiplication using Intel MKL
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, a, n, b, k, 0.0, c, k);

    // Copy the result back to the matrix
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            C[i][j] = c[i * k + j];

    mkl_free(a);
    mkl_free(b);
    mkl_free(c);
    return C;
}

int main(int argc, char const *argv[]) {
    int m, n, k;
    std::vector<std::vector<float>> A, B, C;

    // Read input
    if (AUTO_GENERATE_TEST_CASES) {
        m = n = k = 512;
        A = std::vector<std::vector<float>>(m, std::vector<float>(n));
        B = std::vector<std::vector<float>>(n, std::vector<float>(k));
        C = std::vector<std::vector<float>>(m, std::vector<float>(k));

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                A[i][j] = (float)rand() / RAND_MAX;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                B[i][j] = (float)rand() / RAND_MAX;
    }
    else {
        std::cin >> m >> n >> k;
        A = std::vector<std::vector<float>>(m, std::vector<float>(n));
        B = std::vector<std::vector<float>>(n, std::vector<float>(k));
        C = std::vector<std::vector<float>>(m, std::vector<float>(k));

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                std::cin >> A[i][j];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                std::cin >> B[i][j];
    }


    // Multiply
    std::cerr << "Calculating " << m << "*" << n << "*" << k << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    C = A * B;
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "Multiplication time: " << duration.count() << " ms" << std::endl;
    float flops = (2.0 * m * n * k) / (duration.count() * 1e6);
    std::cerr << "Performance: " << flops << " GFLOPS" << std::endl;

    return 0;
}