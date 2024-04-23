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
set(VALUES 128 256 512 1024 2048)
set(THREADS_VALUES 1 2 4 8 16)
foreach(threads ${THREADS_VALUES})
    foreach(val ${VALUES})
        add_test(NAME Test_${val}_${val}_${val}_${threads} COMMAND OpenMPMatMul ${val} ${val} ${val} ${threads})
    endforeach()
endforeach()