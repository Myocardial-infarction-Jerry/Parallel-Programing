cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Add Intel MKL include and library directories
include_directories(/opt/intel/mkl/include)
link_directories(/opt/intel/mkl/lib/intel64)

# Regular compilation for MatMul
add_executable(MatMul MatMul.cpp)

# Add executable for each .cpp file
add_executable(MatMul_Loop MatMul_Loop.cpp)
add_executable(MatMul_LoopExtended MatMul_LoopExtended.cpp)

# Compilation with optimizations for MatMul
add_executable(MatMul_CompileOptimized MatMul.cpp)
target_compile_options(MatMul_CompileOptimized PRIVATE -O3)

# Compilation with Intel MKL optimizations for MatMul
add_executable(MatMul_MKL MatMul_MKL.cpp)
target_compile_options(MatMul_MKL PRIVATE -O3)
target_link_libraries(MatMul_MKL PRIVATE mkl_rt)