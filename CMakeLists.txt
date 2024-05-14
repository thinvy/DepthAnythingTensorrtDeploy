cmake_minimum_required(VERSION 3.10)
project(DepthAnythingTRTDemo)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 依赖Opencv4
find_package(OpenCV 4 REQUIRED)

# 设置 TensorRT 安装路径 根据当前自身系统环境
# set(TensorRT_ROOT /opt/tensorrt/TensorRT-8.6.0.12)
set(TensorRT_ROOT /opt/TensorRT-8.6.1.6)
# 设置 CUDA 安装路径 根据当前自身系统环境
set(CUDA_ROOT /usr/local/cuda)

# 添加可执行文件
add_executable(DepthAnythingTRTDemo
    main.cpp
    depth_anything_trtruntime/trt_module.cpp
)

# 包含头文件路径
target_include_directories(DepthAnythingTRTDemo PRIVATE 
    ${TensorRT_ROOT}/include
    ${CUDA_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    depth_anything_trtruntime
)

# 链接库文件
target_link_libraries(DepthAnythingTRTDemo PRIVATE
    ${TensorRT_ROOT}/lib/libnvinfer.so
    ${TensorRT_ROOT}/lib/libnvonnxparser.so
    ${CUDA_ROOT}/lib64/libcudart.so
    ${OpenCV_LIBS}
)

# 设置编译选项, 开启o3优化，pre/post process会快一些
target_compile_options(DepthAnythingTRTDemo PRIVATE -Wall -O3)
