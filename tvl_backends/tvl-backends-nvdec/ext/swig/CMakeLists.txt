find_package(SWIG REQUIRED)
include(UseSWIG)

set_property(SOURCE tvlnv.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE tvlnv.i PROPERTY SWIG_MODULE_NAME _tvlnv)
swig_add_library(_tvlnv
    TYPE SHARED
    LANGUAGE python
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/swig
    SOURCES tvlnv.i
)
set_target_properties(_tvlnv PROPERTIES
    OUTPUT_NAME tvlnv
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
)
target_include_directories(_tvlnv
    PRIVATE
    ..
    ../nvidia
    ${CUDA_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(_tvlnv
    PRIVATE
    tvlnv
    ${Python_LIBRARIES}
)

set(RELATIVE_INSTALL_DEST
    "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages"
)

# Install the compiled library file.
set(CMAKE_SHARED_LIBRARY_SUFFIX ".${Python_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(TARGETS _tvlnv
    LIBRARY DESTINATION ${RELATIVE_INSTALL_DEST}
)
# Install the Python file.
install(FILES ${CMAKE_BINARY_DIR}/swig/tvlnv.py
    DESTINATION ${RELATIVE_INSTALL_DEST}
)
