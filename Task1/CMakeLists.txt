cmake_minimum_required(VERSION 3.10)
project(MPIMatMul)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_STANDARD 14)

# BEGIN: Add CUDA support
set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14)

cuda_add_executable(CUDAMatMul CUDAMatMul.cu)
target_link_libraries(CUDAMatMul ${CUDA_LIBRARIES})
# END: Add CUDA support

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(MPIMatMul MPIMatMul.cpp)
