cmake_minimum_required(VERSION 3.10)
project(OpenMPMatMul)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(OpenMPMatMul OpenMPMatMul.cpp)

target_link_libraries(OpenMPMatMul PRIVATE nlohmann_json::nlohmann_json)

# Enable testing
enable_testing()

# Add tests
add_test(NAME Test128 COMMAND OpenMPMatMul 128 128 128)
add_test(NAME Test256 COMMAND OpenMPMatMul 256 256 256)
add_test(NAME Test512 COMMAND OpenMPMatMul 512 512 512)
add_test(NAME Test1024 COMMAND OpenMPMatMul 1024 1024 1024)
add_test(NAME Test2048 COMMAND OpenMPMatMul 2048 2048 2048)