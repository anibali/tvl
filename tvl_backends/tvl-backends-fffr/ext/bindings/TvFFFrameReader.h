#pragma once

#include "MemManager.h"

#include "FfFrameReader.h"

#include <map>
#include <memory>
#include <string>
#include <variant>

class TvFFFrameReader
{
public:
    TvFFFrameReader(
        MemManager* mem_manager, std::string video_file_path, int gpu_index, int out_width = 0, int out_height = 0);
    ~TvFFFrameReader() = default;

    std::string get_filename();
    int get_width();
    int get_height();
    int get_frame_size();
    double get_duration();
    double get_frame_rate();
    int64_t get_number_of_frames();
    void seek(float time_secs);
    uint8_t* const* read_frame();

private:
    std::shared_ptr<FfFrameReader::Stream> _stream = nullptr;
    MemManager* _mem_manager = nullptr;
    std::string _filename;

    uint8_t* allocate(uint32_t size);
    void free(uint8_t* data);

    static std::mutex _mutex;
    static std::map<FfFrameReader::DecoderContext::DecoderOptions, std::shared_ptr<FfFrameReader::DecoderContext>>
        _decoders;
    static std::variant<bool, std::shared_ptr<FfFrameReader::Stream>> getStream(
        const std::string& filename, const FfFrameReader::DecoderContext::DecoderOptions& options) noexcept;
};
