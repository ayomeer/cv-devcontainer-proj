cmake_minimum_required(VERSION 2.8)
project(cppmodule LANGUAGES CXX CUDA)

# Set C++11 as minimum required standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CSS_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add_executable(cppmodule cppmodule.cu)

add_subdirectory(pybind11) # add CMakeLists in pybind11 subdir to the cmake project
pybind11_add_module(cppmodule cppmodule.cu) # add_library() somewhere inside, creates cppmodule target

set_target_properties(cppmodule PROPERTIES CUDA_ARCHITECTURES "86") # see https://developer.nvidia.com/cuda-gpus

# Find and link OpenCV library
find_package(OpenCV REQUIRED)
target_link_libraries(cppmodule PRIVATE ${OpenCV_LIBS}) # OpenCV_LIBS: list of targets to link