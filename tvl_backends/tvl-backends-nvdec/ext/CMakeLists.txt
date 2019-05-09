add_library(tvlnv
    STATIC
    TvlnvFrameReader.cpp
    nvidia/NvDecoder/NvDecoder.cpp
    nvidia/MemManager.cpp
)

# TODO: Error message upon fail to find
find_library(CUDA_LIBRARY cuda)
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)
find_path(AVCODEC_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)
find_path(AVCODEC_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

target_link_directories(tvlnv
    PRIVATE
    ./nvidia/Lib/linux/stubs/x86_64
)

target_include_directories(tvlnv
    PRIVATE
    .
    ./nvidia
    ${CUDA_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIR}
    ${AVUTIL_INCLUDE_DIR}
    ${AVFORMAT_INCLUDE_DIR}
)

target_link_libraries(tvlnv
    PRIVATE
    ${CUDA_LIBRARY}
    nvcuvid
    ${AVCODEC_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${PYTHON_LIBRARIES}
)
