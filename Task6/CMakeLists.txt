cmake_minimum_required(VERSION 3.10)
project(VecAdd)

set(CMAKE_CXX_STANDARD 14)

find_package(nlohmann_json 3.2.0 REQUIRED)

if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -lomp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP REQUIRED)

message(STATUS "OpenMP_C_FLAGS: ${OpenMP_C_FLAGS}")
message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

# Specify the directory of source files
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

# Specify the directory of header files
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# Specify the directory for intermediate files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Create a shared library
add_library(parallel_for SHARED ${SOURCE_DIR}/parallel_for.cpp)
target_include_directories(parallel_for PRIVATE ${INCLUDE_DIR})

# Create executables
add_executable(VecAdd ${SOURCE_DIR}/VecAdd.cpp)
add_executable(MatMul ${SOURCE_DIR}/MatMul.cpp)
add_executable(HeatedPlate ${SOURCE_DIR}/HeatedPlate.cpp)
add_executable(HeatedPlateOpenMP ${SOURCE_DIR}/HeatedPlateOpenMP.cpp)


# Link executables to the shared library and nlohmann_json
target_include_directories(VecAdd PRIVATE ${INCLUDE_DIR})
target_include_directories(MatMul PRIVATE ${INCLUDE_DIR})
target_include_directories(HeatedPlate PRIVATE ${INCLUDE_DIR})
target_link_libraries(VecAdd parallel_for nlohmann_json::nlohmann_json)
target_link_libraries(MatMul parallel_for nlohmann_json::nlohmann_json)
target_link_libraries(HeatedPlate parallel_for nlohmann_json::nlohmann_json)
target_link_libraries(HeatedPlateOpenMP OpenMP::OpenMP_CXX)