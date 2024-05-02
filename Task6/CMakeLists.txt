cmake_minimum_required(VERSION 3.10)
project(PthreadsFor)

set(CMAKE_CXX_STANDARD 14)

# Specify the directory of source files
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

# Specify the directory of header files
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# Specify the directory for intermediate files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Create a shared library
add_library(parallel_for SHARED ${SOURCE_DIR}/parallel_for.cpp)

# Add include directory for the library
target_include_directories(parallel_for PRIVATE ${INCLUDE_DIR})

# Create an executable
add_executable(PthreadsFor ${SOURCE_DIR}/PthreadsFor.cpp)

# Link the executable to the shared library
target_include_directories(PthreadsFor PRIVATE ${INCLUDE_DIR})
target_link_libraries(PthreadsFor parallel_for)
