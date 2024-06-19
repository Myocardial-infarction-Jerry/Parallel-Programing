cmake_minimum_required(VERSION 3.10)
project(YourProjectName)  # Add a project name here for clarity

# Set the C++ standard and enable CUDA language
set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)

# Set the paths for the executable output and source files
set(EXECUTABLE_OUTPUT_PATH bin)
set(SOURCE_DIR src)
set(INCLUDE_DIR include)

# Include the source and include directories
include_directories(${INCLUDE_DIR})

# Include FetchContent for fetching dependencies
include(FetchContent)

# Fetch nlohmann_json
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# Fetch stb
FetchContent_Declare(
  stb
  URL https://github.com/nothings/stb/archive/refs/heads/master.zip
)
FetchContent_MakeAvailable(stb)
include_directories(${stb_SOURCE_DIR})

# Find the required packages
find_package(CUDA REQUIRED)

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

# Gather all source files
file(GLOB CONVOLUTION_SOURCE "${SOURCE_DIR}/img2col.cu" "${SOURCE_DIR}/utils.cpp" "${SOURCE_DIR}/setup.cpp" "${SOURCE_DIR}/Convolution.cpp")

# Create a single executable
add_executable(Convolution ${CONVOLUTION_SOURCE})
add_executable(KernelGenerate ${SOURCE_DIR}/KernelGenerate.cpp)

# Link the libraries
target_link_libraries(Convolution PRIVATE ${CUDA_LIBRARIES} nlohmann_json::nlohmann_json)
