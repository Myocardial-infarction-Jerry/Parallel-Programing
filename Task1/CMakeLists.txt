cmake_minimum_required(VERSION 3.10)
project(MPIMatMul)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(MPIMatMul MPIMatMul.cpp)

# Get the number of available cores
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    # Set the number of MPI processes to the number of available cores
    set(MPIEXEC_NUMPROC_FLAG "-n")
    set(MPIEXEC_MAX_NUMPROCS ${N})
    set(MPIEXEC_PREFLAGS "${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS}")
    set_target_properties(MPIMatMul PROPERTIES
        MPIEXEC_PREFLAGS "${MPIEXEC_PREFLAGS}")
endif()
