#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <complex>
#include <vector>
#include <mpi.h>
#include <unistd.h> 

#define PI 3.141592653589793

void timeStamp() {
#define TIME_SIZE 40

    static char timeBuffer[TIME_SIZE];
    const struct tm *tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(timeBuffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    std::cout << timeBuffer << std::endl;

#undef TIME_SIZE
}

double frand(double &seed) {
    double d2 = 2147483647;
    seed = fmod(16807.0 * seed, d2);
    return (seed - 1) / (d2 - 1);
}

double cpuTime() {
    return (double)clock() / CLOCKS_PER_SEC;
}

int reverse(int n, int ln2) {
    int result = 0;
    for (int i = 0; i < ln2; i++) {
        if (n & (1 << i)) {
            result |= (1 << (ln2 - 1 - i));
        }
    }

    return result;
}

void FFT(std::vector<std::complex<double>> &a, int sign = 1) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = a.size();
    int ln2 = std::log2(n);

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            int rev = reverse(i, ln2);
            if (i < rev)
                std::swap(a[i], a[rev]);
        }
        // std::cout << "Reversed\n";
    }

    for (int s = 1; s <= ln2; ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        std::complex<double> wm(cos(-2 * PI * sign / m), sin(-2 * PI * sign / m));

        MPI_Bcast(a.data(), n, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        for (int j = rank * m; j < n; j += size * m) {
            std::complex<double> w(1, 0);
            for (int k = 0; k < m2; ++k) {
                std::complex<double> t = w * a[j + k + m2];
                std::complex<double> u = a[j + k];
                a[j + k] = u + t;
                a[j + k + m2] = u - t;
                w *= wm;
            }

            MPI_Send(a.data() + j, m, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
        }

        // std::cout << "Rank " << rank << " done.\n";

        if (rank == 0) {
            for (int i = 1; i < size; ++i)
                for (int j = i * m; j < n; j += size * m) {
                    MPI_Recv(a.data() + j, m, MPI_DOUBLE_COMPLEX, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            // std::cout << "s=" << s << " done.\n";
        }
    }

    if (rank == 0 && sign == -1)
        for (auto &x : a)
            x /= n;
}

int main(int argc, char const *argv[]) {
    int rank, size;
    MPI_Init(&argc, (char ***)argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        timeStamp();

        std::cout
            << "\n"
            << "FFT_MPI\n"
            << "  C++ version\n"
            << "\n"
            << "  Demonstrate an implementation of the Fast Fourier Transform\n"
            << "  of a complex data vector.\n";

        std::cout
            << "\n"
            << "  Accuracy check:\n"
            << "\n"
            << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n"
            << "\n"
            << "             N      NITS    Error         Time          Time/Call     MFLOPS\n"
            << "\n";
    }

    static double seed = 331.0;
    int nits = 10000;

    for (int ln2 = 1; ln2 <= 20; ++ln2) {
        int n = 1 << ln2;

        auto timeDomain = std::vector<std::complex<double>>(n);
        auto timeDomainCopy = std::vector<std::complex<double>>(n);

        if (rank == 0) {
            for (int i = 0; i < n; ++i)
                timeDomain[i] = timeDomainCopy[i] = std::complex<double>(i, 0);
        }

        FFT(timeDomain, 1);
        // FFT(timeDomain, -1);

        if (rank == 0) {
            double err = 0.0;

            for (int i = 0; i < n; ++i)
                err += std::norm(timeDomain[i] - timeDomainCopy[i]);

            err = sqrt(err);

            std::cout
                << "  " << std::setw(12) << n
                << "  " << std::setw(8) << nits
                << "  " << std::setw(12) << err;
        }

        auto cTime1 = cpuTime();

        for (int it = 0; it < nits; ++it) {
            // FFT(timeDomain, 1);
            // FFT(timeDomain, -1);
        }

        auto cTime2 = cpuTime();
        auto cTime = cTime2 - cTime1;

        if (rank == 0) {
            auto flops = 2.0 * nits * n * ln2 / cTime / 1.0E+6;
            auto mFlops = 1.0E-6 * flops;

            std::cout
                << "  " << std::setw(12) << cTime
                << "  " << std::setw(12) << cTime / nits * 0.5
                << "  " << std::setw(12) << mFlops
                << "\n";
        }

        if (ln2 % 4 == 0)
            nits = nits / 10;

        if (nits < 1)
            nits = 1;
    }

    if (rank == 0) {
        std::cout
            << "\n"
            << "FFT_MPI:\n"
            << "  Normal end of execution.\n"
            << "\n";
        timeStamp();
    }

    MPI_Finalize();
    return 0;
}
