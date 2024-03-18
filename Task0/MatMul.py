import time
import random

AUTO_GENERATE_TEST_CASES = True

def matmul(A, B):
    m, n, k = len(A), len(A[0]), len(B[0])
    C = [[0 for _ in range(k)] for _ in range(m)]
    for i in range(m):
        for j in range(k):
            for l in range(n):
                C[i][j] += A[i][l] * B[l][j]
    return C

if __name__ == "__main__":
    if AUTO_GENERATE_TEST_CASES:
        m = n = k = 512
        A = [[random.random() for _ in range(n)] for _ in range(m)]
        B = [[random.random() for _ in range(k)] for _ in range(n)]
    else:
        m, n, k = map(int, input().split())
        A = [list(map(float, input().split())) for _ in range(m)]
        B = [list(map(float, input().split())) for _ in range(n)]

    # Multiply
    print(f"Calculating {m}*{n}*{k}")
    start = time.time()
    C = matmul(A, B)
    end = time.time()

    # Calculate the duration
    duration = (end - start) * 1000
    print(f"Multiplication time: %.0lf ms"%(duration))
    flops = (2.0 * m * n * k) / (duration * 1e6)
    print(f"Performance: %.6lf GFLOPS"%(flops))