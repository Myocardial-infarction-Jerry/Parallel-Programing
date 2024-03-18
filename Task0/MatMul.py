import time
import random

def matmul(A, B):
    # Get the dimensions of matrices A and B
    m, n, k = len(A), len(A[0]), len(B[0])
    
    # Initialize matrix C with zeros
    C = [[0 for _ in range(k)] for _ in range(m)]
    
    # Perform matrix multiplication
    for i in range(m):
        for j in range(k):
            for l in range(n):
                C[i][j] += A[i][l] * B[l][j]
    
    return C

if __name__ == "__main__":
    print("Enter m, n, k: ")
    m, n, k = map(int, input().split())
    
    # Generate random matrices A and B
    A = [[random.random() for _ in range(n)] for _ in range(m)]
    B = [[random.random() for _ in range(k)] for _ in range(n)]

    # Multiply matrices A and B
    print(f"Calculating {m}*{n}*{k}")
    start = time.time()
    C = matmul(A, B)
    end = time.time()

    # Calculate the duration of multiplication
    duration = (end - start) * 1000
    print(f"Multiplication time: %.0lf ms"%(duration))
    
    # Calculate the performance in GFLOPS
    flops = (2.0 * m * n * k) / (duration * 1e6)
    print(f"Performance: %.6lf GFLOPS"%(flops))
    
    # Output the matrices A, B, and C to a file
    with open("output.txt", "w") as file:
        file.write("Matrix A:\n")
        for row in A:
            file.write(" ".join(str(element) for element in row))
            file.write("\n")
        file.write("\nMatrix B:\n")
        for row in B:
            file.write(" ".join(str(element) for element in row))
            file.write("\n")
        file.write("\nMatrix C:\n")
        for row in C:
            file.write(" ".join(str(element) for element in row))
            file.write("\n")
    
    print("Result in output.txt")