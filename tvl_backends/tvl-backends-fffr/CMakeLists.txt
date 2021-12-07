cmake_minimum_required(VERSION 3.14.0)
cmake_policy(SET CMP0086 OLD)

project(tvl_backends_fffr
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# NVIDIA Docker images add the CUDA stub library directory to the environment variable
# "LIBRARY_PATH". Here we make CMake aware of this.
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH $ENV{LIBRARY_PATH})

find_package(CUDA)
find_package(Python3 ${PYTHON_VERSION_STRING} EXACT COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python SOABI suffix: ${Python3_SOABI}")

add_subdirectory(ext)
