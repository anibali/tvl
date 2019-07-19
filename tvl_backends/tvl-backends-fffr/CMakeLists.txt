cmake_minimum_required(VERSION 3.13.0)

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
find_package(Python3 COMPONENTS Interpreter Development)

# Get the SOABI suffix for Python native extensions.
# See PEP-3149: https://www.python.org/dev/peps/pep-3149/.
set(_python_code
  "from sysconfig import get_config_var"
  "print(get_config_var('SOABI'))"
)
execute_process(
  COMMAND
  "${Python3_EXECUTABLE}"
  "-c"
  "${_python_code}"
  OUTPUT_VARIABLE Python3_SOABI
  RESULT_VARIABLE _result
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR
    "execute_process(${Python3_EXECUTABLE} -c '${_python_code}') returned "
    "error code ${_result}")
endif()

message(STATUS "Python SOABI suffix: ${Python3_SOABI}")

add_subdirectory(ext)
