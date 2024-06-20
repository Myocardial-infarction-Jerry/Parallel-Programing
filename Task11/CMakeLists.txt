cmake_minimum_required(VERSION 3.10)
project(YourProjectName)  # 设置项目名称

# 设置 C++ 标准并启用 CUDA 语言
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)
enable_language(CUDA)

# 设置可执行文件的输出路径和源文件目录
set(EXECUTABLE_OUTPUT_PATH bin)
set(SOURCE_DIR src)
set(INCLUDE_DIR include)

# 包含源文件和头文件目录
include_directories(${INCLUDE_DIR})

# 包含 FetchContent 以便获取依赖
include(FetchContent)

# 获取 nlohmann_json
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# 获取 stb 图像库
FetchContent_Declare(
  stb
  URL https://github.com/nothings/stb/archive/refs/heads/master.zip
)
FetchContent_MakeAvailable(stb)
include_directories(${stb_SOURCE_DIR})

# 寻找 CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 手动设置 cuDNN 路径
set(CUDNN_INCLUDE_DIR "/usr/include")  # 更新为实际的 include 目录路径，如果不确定可以使用 find 命令查找
set(CUDNN_LIBRARY "/usr/lib/x86_64-linux-gnu/libcudnn.so")
include_directories(${CUDNN_INCLUDE_DIR})
link_directories(/usr/lib/x86_64-linux-gnu)  # 根据 libcudnn.so 的位置设置正确的链接目录

# 设置调试选项
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

# 搜集所有源代码文件
file(GLOB CONVOLUTION_SOURCE "${SOURCE_DIR}/img2col.cu" "${SOURCE_DIR}/utils.cpp" "${SOURCE_DIR}/setup.cpp" "${SOURCE_DIR}/Convolution.cpp")

# 创建三个可执行文件目标
add_executable(Convolution ${CONVOLUTION_SOURCE})
add_executable(ConvolutionCuDNN ${SOURCE_DIR}/ConvolutionCuDNN.cpp ${SOURCE_DIR}/utils.cpp ${SOURCE_DIR}/setup.cpp)
add_executable(KernelGenerate ${SOURCE_DIR}/KernelGenerate.cpp)

# 链接库
target_link_libraries(Convolution PRIVATE ${CUDA_LIBRARIES} nlohmann_json::nlohmann_json)
target_link_libraries(ConvolutionCuDNN PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY} nlohmann_json::nlohmann_json)
