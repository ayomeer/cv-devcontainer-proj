cmake_minimum_required(VERSION 2.8)
project(cppmodule LANGUAGES CXX CUDA)

# Set C++11 as minimum required standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CSS_STANDARD_REQUIRED ON)

#set(THREADS_PREFER_PTHREAD_FLAG ON) # use the -pthread compiler and linker flag
#set(CMAKE_THREAD_PREFER_PTHREAD ON) # use pthread library if multiple thread libraries available

# add_executable(cppmodule cppmodule.cu)

add_subdirectory(pybind11) # add CMakeLists in pybind11 subdir to the cmake project
pybind11_add_module(cppmodule cppmodule.cu) # add_library() somewhere inside, creates cppmodule target

set_target_properties(cppmodule PROPERTIES CUDA_ARCHITECTURES "61") # compute capability version nvidia quadro is 7.5 --> "75"

# Find and link OpenCV library
find_package(OpenCV REQUIRED)
target_link_libraries(cppmodule PRIVATE ${OpenCV_LIBS}) # OpenCV_LIBS: list of targets to link

