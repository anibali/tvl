# Build C++ bridging code.

add_library(pyfffr
    STATIC
    TvFFFrameReader.cpp
)

target_link_directories(pyfffr
    PRIVATE
    tvl_backends_fffr::FfFrameReader
)

target_include_directories(pyfffr
    PRIVATE
    .
    ../FFFrameReader/include
    ${FfFrameReader_BINARY_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Build SWIG bindings.

find_package(SWIG REQUIRED)
include(UseSWIG)

set_property(SOURCE pyfffr.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE pyfffr.i PROPERTY SWIG_MODULE_NAME _pyfffr)
swig_add_library(_pyfffr
    TYPE SHARED
    LANGUAGE python
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/swig
    SOURCES pyfffr.i
)
set_target_properties(_pyfffr PROPERTIES
    OUTPUT_NAME pyfffr
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
)
target_include_directories(_pyfffr
    PRIVATE
    .
    ../FFFrameReader/include
    ${FfFrameReader_BINARY_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(_pyfffr
    PRIVATE
    pyfffr
    FfFrameReader
    ${PYTHON_LIBRARIES}
)

set(RELATIVE_INSTALL_DEST
    "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages"
)

# Install the compiled library file.
set(CMAKE_SHARED_LIBRARY_SUFFIX ".${Python_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(TARGETS _pyfffr
    LIBRARY DESTINATION ${RELATIVE_INSTALL_DEST}
)
# Install the Python file.
install(FILES ${CMAKE_BINARY_DIR}/swig/pyfffr.py
    DESTINATION ${RELATIVE_INSTALL_DEST}
)
