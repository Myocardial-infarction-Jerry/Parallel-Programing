#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 数据总量和每个进程的数据量
    const int N = 100;
    int local_n = N / size;

    // 主进程准备数据
    std::vector<int> data;
    if (rank == 0) {
        data.resize(N);
        for (int i = 0; i < N; i++) {
            data[i] = i;
        }
    }

    // 每个进程的接收缓冲区
    std::vector<int> local_data(local_n);

    // 分发数据
    MPI_Scatter(data.data(), local_n, MPI_INT,
        local_data.data(), local_n, MPI_INT,
        0, MPI_COMM_WORLD);

    // 输出每个进程接收的数据
    std::cout << "Rank " << rank << " received:";
    for (int i = 0; i < local_n; i++) {
        std::cout << " " << local_data[i];
    }
    std::cout << std::endl;

    MPI_Finalize();
    return 0;
}
