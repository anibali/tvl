#pragma once

#include <cuda.h>

#include "NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"

#include "MemManager.h"

class TvlnvFrameReader
{
public:
    TvlnvFrameReader(MemManager* mem_manager, std::string video_file_path);
    ~TvlnvFrameReader();

    std::string get_filename();
    uint8_t* read_frame();
    void read_frames();

private:
    MemManager* _mem_manager;
    std::string _filename;
    CUcontext _cu_context = NULL;
    FFmpegDemuxer* _demuxer;
    NvDecoder* _decoder;
};
