#pragma once

#include "FFFrameReader.h"
#include "ImageAllocator.h"

#include <cuda.h>
#include <map>
#include <memory>
#include <string>

class TvFFFrameReader
{
public:
    TvFFFrameReader(ImageAllocator* image_allocator, const std::string& filename, int gpu_index, int out_width = 0,
        int out_height = 0, int seek_threshold = 0, int buffer_length = 8);
    ~TvFFFrameReader() = default;

    std::string get_filename() const;
    int get_width() const;
    int get_height() const;
    double get_duration() const;
    double get_frame_rate() const;
    int64_t get_number_of_frames() const;
    void seek(float time_secs);
    void seek_frame(int frame_index);

    uint8_t* read_frame();
    int64_t read_frames_by_index(int64_t* indices, int n_frames, uint8_t** frames);

private:
    static std::map<int, std::shared_ptr<std::remove_pointer<CUcontext>::type>> _contexts;
    std::shared_ptr<Ffr::Stream> _stream = nullptr;
    ImageAllocator* _image_allocator = nullptr;
    std::string _filename;
    Ffr::PixelFormat _pixel_format;

    static bool init_context(int gpu_index);
    
    uint8_t* convert_frame(const std::shared_ptr<Ffr::Frame>& frame, bool async);
};
