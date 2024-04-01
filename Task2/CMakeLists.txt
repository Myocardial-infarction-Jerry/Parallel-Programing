cmake_minimum_required(VERSION 3.10)
project(MPIMatMul)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(MPIMatMul MPIMatMul.cpp)
target_link_libraries(MPIMatMul nlohmann_json::nlohmann_json)