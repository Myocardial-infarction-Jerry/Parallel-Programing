#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

// Function to perform matrix multiplication
std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B) {
    int m = A.size(), n = A[0].size(), k = B[0].size();
    std::vector<std::vector<float>> C(m, std::vector<float>(k, 0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int l = 0;
            for (; l < n - 3; l += 4) {
                C[i][j] += A[i][l] * B[l][j];
                C[i][j] += A[i][l + 1] * B[l + 1][j];
                C[i][j] += A[i][l + 2] * B[l + 2][j];
                C[i][j] += A[i][l + 3] * B[l + 3][j];
            }
            for (; l < n; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }

    return C;
}

// Function to print matrix
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
    std::cerr << "Enter m, n, k: ";
    int m, n, k;
    std::cin >> m >> n >> k;

    A = std::vector<std::vector<float>>(m, std::vector<float>(n));
    B = std::vector<std::vector<float>>(n, std::vector<float>(k));
    C = std::vector<std::vector<float>>(m, std::vector<float>(k));

    // Generate random values for matrix A
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = (float)rand() / RAND_MAX;

    // Generate random values for matrix B
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            B[i][j] = (float)rand() / RAND_MAX;

    // Multiply matrices A and B
    std::cerr << "Calculating " << m << "*" << n << "*" << k << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    C = A * B;
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "Multiplication time: " << duration.count() << " ms" << std::endl;
    float flops = (2.0 * m * n * k) / (duration.count() * 1e6);
    std::cerr << "Performance: " << flops << " GFLOPS" << std::endl;

    // Output result to a file
    std::ofstream out("output.txt");
    out << "Matrix A:" << std::endl << A << std::endl;
    out << "Matrix B:" << std::endl << B << std::endl;
    out << "Matrix C:" << std::endl << C << std::endl;
    std::cerr << "Result in output.txt" << std::endl;

    return 0;
}
