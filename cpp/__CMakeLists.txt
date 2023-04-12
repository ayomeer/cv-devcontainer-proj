cmake_minimum_required(VERSION 3.11)
project(hello LANGUAGES CXX CUDA)
add_executable(hello cuda.cu)
set_target_properties(hello PROPERTIES CUDA_ARCHITECTURES "61") # compute capability version nvidia quadro is 7.5 --> "75"

