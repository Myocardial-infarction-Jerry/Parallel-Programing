cmake_minimum_required(VERSION 3.10)
project(PthreadEquation)

set(CMAKE_CXX_STANDARD 14)

add_executable(PthreadEquation PthreadEquation.cpp)
add_executable(PthreadMonteCarlo PthreadMonteCarlo.cpp)

find_package(Threads REQUIRED)
target_link_libraries(PthreadEquation Threads::Threads)
target_link_libraries(PthreadMonteCarlo Threads::Threads)