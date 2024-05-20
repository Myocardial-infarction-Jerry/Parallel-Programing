cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 14)

# Specify the directory where the source files are
set(SOURCE_DIR src)

# Specify the directory where the executable files will be generated
set(EXECUTABLE_OUTPUT_PATH bin)

# Get all source files
file(GLOB SOURCES "${SOURCE_DIR}/*.cpp")

# For each source file, create an executable
foreach(source ${SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
endforeach()