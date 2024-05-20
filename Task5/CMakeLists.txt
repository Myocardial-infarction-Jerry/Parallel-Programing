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