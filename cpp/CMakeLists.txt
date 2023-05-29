cmake_minimum_required(VERSION 2.8)
project(cppmodule)

# Set C++11 as minimum required standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CSS_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON) # use the -pthread compiler and linker flag
set(CMAKE_THREAD_PREFER_PTHREAD ON) # use pthread library if multiple thread libraries available
find_package(Threads REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(cppmodule cppmodule.cpp) # creates library target like add_library() would

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(cppmodule PRIVATE ${OpenCV_LIBS})
target_link_libraries(cppmodule PRIVATE Threads::Threads) 