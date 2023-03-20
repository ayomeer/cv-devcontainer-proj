cmake_minimum_required(VERSION 2.8)
project( cppmodule )

add_subdirectory(pybind11)
pybind11_add_module(cppmodule cppmodule.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# add_executable( cppmodule cppmodule.cpp )
target_link_libraries( cppmodule ${OpenCV_LIBS} )