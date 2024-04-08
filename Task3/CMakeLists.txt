cmake_minimum_required(VERSION 3.10)
project(PthreadProject)

set(CMAKE_CXX_STANDARD 17)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Threads REQUIRED)

add_executable(PthreadMatMul PthreadMatMul.cpp)
add_executable(PthreadVecSum PthreadVecSum.cpp)

target_link_libraries(PthreadMatMul PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
target_link_libraries(PthreadVecSum PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
