cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Add Intel MKL include and library directories
include_directories(/opt/intel/mkl/include)
link_directories(/opt/intel/mkl/lib/intel64)

# Set source files directory
set(SOURCE_DIR src)

# Set executable output directory
set(EXECUTABLE_OUTPUT_PATH bin)

# Regular compilation for MatMul
add_executable(MatMul ${SOURCE_DIR}/MatMul.cpp)
add_executable(MatMul_Loop ${SOURCE_DIR}/MatMul_Loop.cpp)
add_executable(MatMul_LoopExtended ${SOURCE_DIR}/MatMul_LoopExtended.cpp)

# Compile MatMul_CompileOptimized with optimization flags
add_executable(MatMul_CompileOptimized ${SOURCE_DIR}/MatMul.cpp)
target_compile_options(MatMul_CompileOptimized PRIVATE -O3)

# Check if MKL library exists
find_library(MKL_LIBRARY mkl_rt)
if(MKL_LIBRARY)
    # Compilation with Intel MKL optimizations for MatMul
    add_executable(MatMul_MKL ${SOURCE_DIR}/MatMul_MKL.cpp)
    target_compile_options(MatMul_MKL PRIVATE -O3)
    target_link_libraries(MatMul_MKL PRIVATE ${MKL_LIBRARY})
else()
    message(WARNING "MKL library not found. MatMul_MKL will not be compiled.")
endif()
