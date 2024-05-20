#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

const double PI = 3.141592653589793238460;

// 位逆转函数，用于重新排列输入数组的顺序
int reverse(int num, int log2n) {
    int result = 0;
    for (int i = 0; i < log2n; i++) {
        if (num & (1 << i)) {
            result |= (1 << (log2n - 1 - i));
        }
    }
    return result;
}

// FFT和IFFT的统一实现，通过sign参数控制
void fft(std::vector<std::complex<double>> &a, double sign = 1.0) {
    int n = a.size();
    int log2n = std::log2(n);

    // 重新排列数组元素的顺序
    for (int i = 0; i < n; ++i) {
        int rev = reverse(i, log2n);
        if (i < rev)
            std::swap(a[i], a[rev]);
    }

    // 主FFT/IFFT循环
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s; // 当前级别的点数
        int m2 = m >> 1; // 子问题大小
        std::complex<double> wm(cos(-2 * PI * sign / m), sin(-2 * PI * sign / m)); // W_m, sign控制方向

        for (int j = 0; j < n; j += m) {
            std::complex<double> w(1, 0);
            for (int k = 0; k < m2; ++k) {
                std::complex<double> t = w * a[j + k + m2];
                std::complex<double> u = a[j + k];
                a[j + k] = u + t;
                a[j + k + m2] = u - t;
                w *= wm;
            }
        }
    }

    // 如果是IFFT，则进行归一化
    if (sign == -1.0) {
        for (auto &x : a) {
            x /= n;
        }
    }
}

// 主函数，演示FFT和IFFT的使用
int main() {
    std::vector<std::complex<double>> data = {
        {0, 0}, {1, 0}, {2, 0}, {3, 0},
        {4, 0}, {5, 0}, {6, 0}, {7, 0}
    };

    std::cout << "Original Data:\n";
    for (const auto &d : data) {
        std::cout << d << ' ';
    }
    std::cout << "\n\n";

    fft(data, 1.0); // 正向FFT
    std::cout << "After FFT:\n";
    for (const auto &d : data) {
        std::cout << d << ' ';
    }
    std::cout << "\n\n";

    fft(data, -1.0); // 逆向FFT
    std::cout << "After IFFT:\n";
    for (const auto &d : data) {
        std::cout << d << ' ';
    }
    std::cout << "\n";

    return 0;
}
