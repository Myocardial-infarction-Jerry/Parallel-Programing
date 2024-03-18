#include <iostream>
#include <vector>
#include <chrono>

#define AUTO_GENERATE_TEST_CASES 1

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
