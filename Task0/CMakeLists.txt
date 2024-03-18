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

# Check if MKL library exists
find_library(MKL_LIBRARY mkl_rt)
if(MKL_LIBRARY)
    # Compilation with Intel MKL optimizations for MatMul
    add_executable(MatMul_MKL MatMul_MKL.cpp)
    target_compile_options(MatMul_MKL PRIVATE -O3)
    target_link_libraries(MatMul_MKL PRIVATE ${MKL_LIBRARY})
else()
    message(WARNING "MKL library not found. MatMul_MKL will not be compiled.")
endif()
