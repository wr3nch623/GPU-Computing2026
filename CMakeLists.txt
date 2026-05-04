cmake_minimum_required(VERSION 3.10.0)
project(GPU-Computing2026 VERSION 0.1.0 LANGUAGES CXX CUDA)
# Change for relevant CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

add_executable(GPU-Computing2026 main.cpp)


include(CTest)
enable_testing()

