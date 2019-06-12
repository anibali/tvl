#pragma once

#include "FFFrameReader.h"
#include "ImageAllocator.h"

#include <memory>
#include <string>

class TvFFFrameReader
{
public:
    TvFFFrameReader(
        ImageAllocator* image_allocator, const std::string& filename, int gpu_index,
        int out_width = 0, int out_height = 0);
    ~TvFFFrameReader() = default;

    std::string get_filename();
    int get_width();
    int get_height();
    double get_duration();
    double get_frame_rate();
    int64_t get_number_of_frames();
    void seek(float time_secs);

    uint8_t* read_frame();

private:
    std::shared_ptr<Ffr::Stream> _stream = nullptr;
    ImageAllocator* _image_allocator = nullptr;
    std::string _filename;
    Ffr::PixelFormat _pixel_format;
};
