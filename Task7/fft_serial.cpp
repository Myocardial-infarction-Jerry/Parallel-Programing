#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>

#define PI 3.141592653589793

void timestamp() {
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

double ggl(double &seed) {
    double d2 = 2147483647;
    seed = fmod(16807.0 * seed, d2);
    return (seed - 1) / (d2 - 1);
}

void cffti(int n, double w[]) {
    auto n2 = n / 2;
    auto aw = 2.0 * PI / (double)n;

    for (int i = 0; i < n2; i++) {
        auto arg = aw * (double)i;
        w[i * 2] = cos(arg);
        w[i * 2 + 1] = sin(arg);
    }
}

void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn) {
    double wjw[2];
    auto mj2 = 2 * mj;
    auto lj = n / mj2;

    for (int j = 0; j < lj; j++) {
        auto jw = j * mj;
        auto ja = jw;
        auto jb = ja;
        auto jc = j * mj2;
        auto jd = jc;

        wjw[0] = w[jw * 2 + 0];
        wjw[1] = w[jw * 2 + 1];

        if (sgn < 0.0) {
            wjw[1] = -wjw[1];
        }

        for (int k = 0; k < mj; k++) {
            c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
            c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

            auto ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
            auto ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

            d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
            d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
        }
    }
}

void ccopy(int n, double x[], double y[]) {
    for (int i = 0; i < n; i++) {
        y[i * 2 + 0] = x[i * 2 + 0];
        y[i * 2 + 1] = x[i * 2 + 1];
    }
}

void cfft2(int n, double x[], double y[], double w[], double sgn) {
    auto m = (int)(log((double)n) / log(1.99));
    auto mj = 1;

    auto tgle = 1;
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

    if (n == 2) {
        return;
    }

    for (int j = 0; j < m - 2; j++) {
        mj = mj * 2;
        if (tgle) {
            step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0], &x[0 * 2 + 0], &x[mj * 2 + 0], w, sgn);
            tgle = 0;
        }
        else {
            step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
            tgle = 1;
        }
    }
    //
    //  Last pass thru data: move y to x if needed 
    //
    if (tgle) {
        ccopy(n, y, x);
    }

    mj = n / 2;
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
}

double cpu_time() { return (double)clock() / (double)CLOCKS_PER_SEC; }

int main(int argc, char const *argv[]) {
    timestamp();
    std::cout
        << "\n"
        << "FFT_SERIAL\n"
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

    static double seed = 331.0;
    int n = 1;
    int nits = 10000;

    for (int ln2 = 1; ln2 <= 20; ++ln2) {
        n <<= 1;

        auto w = new double[n];
        auto x = new double[2 * n];
        auto y = new double[2 * n];
        auto z = new double[2 * n];

        bool first = 1;

        for (int icase = 0; icase < 2; ++icase) {
            for (int i = 0; i < 2 * n; i += 2) {
                x[i] = z[i] = first ? ggl(seed) : 0;
                x[i + 1] = z[i + 1] = first ? ggl(seed) : 0;
            }

            cffti(n, w);

            if (first) {
                auto sgn = +1.0;
                cfft2(n, x, y, w, sgn);
                sgn = -1.0;
                cfft2(n, y, x, w, sgn);
                // 
                //  Results should be same as initial multiplied by N.
                //
                auto fnm1 = 1.0 / (double)n;
                auto error = 0.0;

                for (int i = 0; i < 2 * n; i += 2) {
                    error = error
                        + pow(z[i] - fnm1 * x[i], 2)
                        + pow(z[i + 1] - fnm1 * x[i + 1], 2);
                }

                error = sqrt(fnm1 * error);
                std::cout
                    << "  " << std::setw(12) << n
                    << "  " << std::setw(8) << nits
                    << "  " << std::setw(12) << error;
                first = 0;
            }
            else {
                auto ctime1 = cpu_time();

                for (int it = 0; it < nits; it++) {
                    auto sgn = +1.0;
                    cfft2(n, x, y, w, sgn);
                    sgn = -1.0;
                    cfft2(n, y, x, w, sgn);
                }

                auto ctime2 = cpu_time();
                auto ctime = ctime2 - ctime1;

                auto flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);

                auto mflops = flops / 1.0E+06 / ctime;

                std::cout
                    << "  " << std::setw(12) << ctime
                    << "  " << std::setw(12) << ctime / (double)(2 * nits)
                    << "  " << std::setw(12) << mflops << "\n";
            }
        }

        if ((ln2 % 4) == 0)
            nits = nits / 10;

        if (nits < 1)
            nits = 1;

        delete[] w;
        delete[] x;
        delete[] y;
        delete[] z;
    }

    std::cout
        << "\n"
        << "FFT_MPI:\n"
        << "  Normal end of execution.\n"
        << "\n";
    timestamp();
    return 0;
}

