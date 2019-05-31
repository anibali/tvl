#pragma once

#include "FFFrameReader.h"
#include "MemManager.h"

#include <memory>
#include <string>

class TvFFFrameReader
{
public:
    TvFFFrameReader(
        MemManager* mem_manager, const std::string& filename, int gpu_index, int out_width = 0, int out_height = 0);
    ~TvFFFrameReader() = default;

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
    std::shared_ptr<Ffr::Stream> _stream = nullptr;
    MemManager* _mem_manager = nullptr;
    std::string _filename;
};
