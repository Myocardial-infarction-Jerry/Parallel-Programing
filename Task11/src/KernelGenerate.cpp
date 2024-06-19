#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <random>

class KernelGenerator {
public:
    void generateKernel(int height, int width, int channels, const std::string &directory) {
        // Generate random kernel data
        float *data = new float[height * width * channels];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < height * width * channels; ++i) {
            data[i] = static_cast<float>(dis(gen));
        }

        // Generate file name with current date and time
        std::string filename = generateFilename(height, width, channels, directory);

        // Write data to binary file
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error creating kernel file: " << filename << std::endl;
            delete[] data;
            return;
        }

        file.write((char *)&height, sizeof(int));
        file.write((char *)&width, sizeof(int));
        file.write((char *)&channels, sizeof(int));
        file.write((char *)data, height * width * channels * sizeof(float));
        file.close();

        std::cout << "Generated kernel file: " << filename << std::endl;

        delete[] data;
    }

private:
    std::string generateFilename(int height, int width, int channels, const std::string &directory) {
        // Get current time
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        // Create time string
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        std::string timeStr = oss.str();

        // Create filename
        std::ostringstream filename;
        filename << directory << "/kernel_" << width << "x" << height << "x" << channels << "_" << timeStr << ".dat";

        return filename.str();
    }
};

int main(int argc, char const *argv[]) {
    KernelGenerator kg;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <height> <width> <channels>" << std::endl;
        return 1;
    }

    int height = std::stoi(argv[1]);
    int width = std::stoi(argv[2]);
    int channels = std::stoi(argv[3]);

    kg.generateKernel(height, width, channels, "res");

    return 0;
}
