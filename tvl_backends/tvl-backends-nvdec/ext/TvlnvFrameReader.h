#pragma once

#include <cuda.h>
#include <queue>

#include "NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"

#include "MemManager.h"

class TvlnvFrameReader
{
public:
    TvlnvFrameReader(MemManager* mem_manager, std::string video_file_path, int gpu_index,
                     int out_width=0, int out_height=0);
    ~TvlnvFrameReader();

    std::string get_filename();
    int get_width();
    int get_height();
    int get_frame_size();
    double get_duration();
    double get_frame_rate();
    int64_t get_number_of_frames();
    void seek(float time_secs);
    uint8_t* read_frame();

private:
    void _init_decoder() {
        _decoder = new NvDecoder(_cu_context, _demuxer->GetWidth(), _demuxer->GetHeight(),
                                 _mem_manager, FFmpeg2NvCodecId(_demuxer->GetVideoCodec()),
                                 NULL, false, NULL, _resize_dim);
    }

    CUdevice _cu_device;
    MemManager* _mem_manager;
    std::string _filename;
    CUcontext _cu_context = NULL;
    FFmpegDemuxer* _demuxer;
    NvDecoder* _decoder;
    std::queue<uint8_t*> frame_buf;
    int64_t _seek_pts = AV_NOPTS_VALUE;
    Dim* _resize_dim = NULL;
};
