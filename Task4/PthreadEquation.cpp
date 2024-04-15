#include <iostream>
#include <cmath>
#include <pthread.h>
#include <ctime>
#include <complex>

struct QuadraticEquation {
    double a, b, c;
    double discriminant;
    std::complex<double> root1, root2;
    bool discriminantComputed;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

void *computeDiscriminant(void *arg) {
    QuadraticEquation *eq = (QuadraticEquation *)arg;
    pthread_mutex_lock(&eq->mutex);

    eq->discriminant = eq->b * eq->b - 4 * eq->a * eq->c;
    eq->discriminantComputed = true;

    pthread_cond_signal(&eq->cond);
    pthread_mutex_unlock(&eq->mutex);
    return nullptr;
}

void *computeRoots(void *arg) {
    QuadraticEquation *eq = (QuadraticEquation *)arg;
    pthread_mutex_lock(&eq->mutex);

    while (!eq->discriminantComputed)
        pthread_cond_wait(&eq->cond, &eq->mutex);

    if (eq->discriminant >= 0) {
        eq->root1 = (-eq->b + sqrt(eq->discriminant)) / (2 * eq->a);
        eq->root2 = (-eq->b - sqrt(eq->discriminant)) / (2 * eq->a);
    }
    else {
        double realPart = -eq->b / (2 * eq->a);
        double imagPart = sqrt(-eq->discriminant) / (2 * eq->a);
        eq->root1 = std::complex<double>(realPart, imagPart);
        eq->root2 = std::complex<double>(realPart, -imagPart);
    }

    pthread_mutex_unlock(&eq->mutex);
    return nullptr;
}

std::ostream &operator<<(std::ostream &out, const std::complex<double> &c) {
    out << c.real();
    if (fabs(c.imag()) > 1E-6) {
        if (c.imag() >= 0)
            out << "+";
        out << c.imag() << "i";
    }
    return out;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <a> <b> <c>" << std::endl;
        return 1;
    }

    double a = atof(argv[1]), b = atof(argv[2]), c = atof(argv[3]);
    QuadraticEquation eq = { a, b, c, 0, std::complex<double>(0, 0), std::complex<double>(0, 0), false, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER };

    clock_t start = clock();

    pthread_t thread1, thread2;
    pthread_create(&thread1, nullptr, computeDiscriminant, &eq);
    pthread_create(&thread2, nullptr, computeRoots, &eq);

    pthread_join(thread1, nullptr);
    pthread_join(thread2, nullptr);

    clock_t end = clock();
    double elapsedTime = double(end - start) / CLOCKS_PER_SEC;

    std::cerr << "Roots: " << eq.root1 << ", " << eq.root2 << std::endl;
    std::cerr << "Running time: " << elapsedTime << " seconds" << std::endl;

    return 0;
}