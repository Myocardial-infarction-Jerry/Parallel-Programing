cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)

# Enable CUDA language
enable_language(CUDA)

# Specify the directory where the source files are
set(SOURCE_DIR src)

# Specify the directory where the executable files will be generated
set(EXECUTABLE_OUTPUT_PATH bin)

# Get all source files
file(GLOB SOURCES "${SOURCE_DIR}/*.cu" "${SOURCE_DIR}/*.cpp")

# Find CUDA and MPI packages
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

# For each source file, create an executable
foreach(source ${SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    # If the source file is a .cpp file, use MPI compiler
    if(${source} MATCHES ".cpp$")
        set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(${name} MPI::MPI_CXX)
    endif()
    # Link CUDA libraries
    if(${source} MATCHES ".cu$")
        target_link_libraries(${name} ${CUDA_LIBRARIES})
    endif()
endforeach()