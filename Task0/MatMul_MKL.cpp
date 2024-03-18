#include <iostream>
#include <vector>
#include <chrono>
#include <mkl.h> 
#include <fstream>

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

std::ostream &operator<<(std::ostream &out, const std::vector<std::vector<float>> &mat) {
    for (auto &row : mat) {
        for (auto &val : row)
            out << val << " ";
        out << std::endl;
    }

    return out;
}

int main(int argc, char const *argv[]) {
    std::vector<std::vector<float>> A, B, C;

    // Read input
    int m, n, k;
    std::cin >> m >> n >> k;

    A = std::vector<std::vector<float>>(m, std::vector<float>(n));
    B = std::vector<std::vector<float>>(n, std::vector<float>(k));
    C = std::vector<std::vector<float>>(m, std::vector<float>(k));

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = (float)rand() / RAND_MAX;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            B[i][j] = (float)rand() / RAND_MAX;

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

    // Output result
    std::ofstream out("output.txt");
    out << "Matrix A:" << std::endl << A << std::endl;
    out << "Matrix B:" << std::endl << B << std::endl;
    out << "Matrix C:" << std::endl << C << std::endl;
    std::cerr << "Result in output.txt" << std::endl;

    return 0;
}