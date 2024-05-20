cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# Specify the directory where the source files are
set(SOURCE_DIR src)

# Specify the directory where the executable files will be generated
set(EXECUTABLE_OUTPUT_PATH bin)

# Specify the directory where the header files are
set(INCLUDE_DIR include)

# Add the include directory to the compiler's header file search path
include_directories(${INCLUDE_DIR})

# Get all source files
file(GLOB SOURCES "${SOURCE_DIR}/*.cpp")

# For each source file, create an executable
foreach(source ${SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
endforeach()
