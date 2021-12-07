cmake_minimum_required(VERSION 3.14.0)
cmake_policy(SET CMP0086 OLD)

project(tvl_backends_nvdec
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA)
find_package(Python3 ${PYTHON_VERSION_STRING} EXACT COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python SOABI suffix: ${Python3_SOABI}")

add_subdirectory(ext)
add_subdirectory(ext/swig)
