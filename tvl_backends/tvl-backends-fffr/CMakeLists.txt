cmake_minimum_required(VERSION 3.17.0)
cmake_policy(SET CMP0086 OLD)

project(tvl_backends_fffr
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# NVIDIA Docker images add the CUDA stub library directory to the environment variable
# "LIBRARY_PATH". Here we make CMake aware of this.
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH $ENV{LIBRARY_PATH})

find_package(CUDA)

if(SKBUILD)
    # Use Python location hints supplied by scikit-build.
    # See: https://github.com/scikit-build/scikit-build/issues/506
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
    set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
    set(Python_LIBRARY "${PYTHON_LIBRARY}")
    find_package(Python ${PYTHON_VERSION_STRING} EXACT COMPONENTS Interpreter Development REQUIRED)
else()
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
endif()
message(STATUS "Python SOABI suffix: ${Python_SOABI}")

add_subdirectory(ext)
