cmake_minimum_required(VERSION 3.17.0)
cmake_policy(SET CMP0086 OLD)

project(tvl_backends_nvdec
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
add_subdirectory(ext/swig)
