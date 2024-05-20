cmake_minimum_required(VERSION 3.10)

# Set the project name
project(FFTProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the global C++ compiler to clang++
set(CMAKE_CXX_COMPILER clang++)

# Add the fft_serial executable
add_executable(fft_serial src/fft_serial.cpp)

# Add a custom command to compile fft_mpi.cpp with mpicxx
add_custom_command(
    OUTPUT fft_mpi
    COMMAND mpicxx -std=c++11 -o ${CMAKE_BINARY_DIR}/bin/fft_mpi src/fft_mpi.cpp
    DEPENDS src/fft_mpi.cpp
)

# Add a custom target that depends on the custom command
add_custom_target(
    fft_mpi_target ALL
    DEPENDS fft_mpi
)

# Set the output directory for the fft_serial executable to be the bin directory
set_target_properties(fft_serial PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)