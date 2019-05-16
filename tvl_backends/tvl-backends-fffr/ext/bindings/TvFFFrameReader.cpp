#include "TvFFFrameReader.h"

#include <libavutil/avutil.h>

std::mutex TvFFFrameReader::_mutex;
std::map<FfFrameReader::DecoderContext::DecoderOptions, std::shared_ptr<FfFrameReader::DecoderContext>> TvFFFrameReader::_decoders;

std::variant<bool, std::shared_ptr<FfFrameReader::Stream>> TvFFFrameReader::getStream(
    const std::string& filename, const FfFrameReader::DecoderContext::DecoderOptions& options) noexcept
{
    try {
        std::lock_guard<std::mutex> lock(TvFFFrameReader::_mutex);
        // Check if a manager already registered for type
        const auto foundManager = _decoders.find(options);
        std::shared_ptr<FfFrameReader::DecoderContext> found;
        if (foundManager == _decoders.end()) {
            _decoders.emplace(options, std::make_shared<FfFrameReader::DecoderContext>(options));
            found = _decoders.find(options)->second;
        } else {
            found = foundManager->second;
        }

        // Create a new stream using the requested manager
        const auto newStream = found->getStream(filename);
        if (newStream.index() != 0) {
            return std::get<1>(newStream);
        }
    } catch (...) {
    }
    return false;
}

TvFFFrameReader::TvFFFrameReader(
    MemManager* mem_manager, const std::string filename, const int gpu_index, const int out_width, const int out_height)
    : _mem_manager(mem_manager)
    , _filename(filename)
{
    FfFrameReader::DecoderContext::DecodeType decode_type = FfFrameReader::DecoderContext::DecodeType::Software;
    if(gpu_index >= 0) {
        decode_type = FfFrameReader::DecoderContext::DecodeType::Nvdec;
    }

    FfFrameReader::DecoderContext::DecoderOptions options(decode_type);
    options.m_device = gpu_index;
    const std::function<uint8_t*(uint32_t)> allocator =
        std::bind(&TvFFFrameReader::allocate, this, std::placeholders::_1);
    const std::function<void(uint8_t*)> free = std::bind(&TvFFFrameReader::free, this, std::placeholders::_1);
    options.m_allocator = std::optional<FfFrameReader::DecoderContext::DecoderOptions::Allocator>({allocator, free});
    const auto ret = getStream(filename, options);
    if (ret.index() == 0) {
        throw;
    }
    _stream = std::get<1>(ret);
}

std::string TvFFFrameReader::get_filename()
{
    return _filename;
}

int TvFFFrameReader::get_width()
{
    return _stream->getWidth();
}

int TvFFFrameReader::get_height()
{
    return _stream->getHeight();
}

int TvFFFrameReader::get_frame_size()
{
    return _stream->getFrameSize();
}

double TvFFFrameReader::get_duration()
{
    return _stream->getDuration() / (double)AV_TIME_BASE;
}

double TvFFFrameReader::get_frame_rate()
{
    return _stream->getFrameRate();
}

int64_t TvFFFrameReader::get_number_of_frames()
{
    return _stream->getTotalFrames();
}

void TvFFFrameReader::seek(const float time_secs)
{
    const bool ret = _stream->seek(time_secs * AV_TIME_BASE);
    if (!ret) {
        throw;
    }
}

uint8_t* const* TvFFFrameReader::read_frame()
{
    const auto ret = _stream->getNextFrame();
    if (ret.index() == 0) {
        return nullptr;
    }
    const auto frame = std::get<1>(ret);
    return frame->getFrameData();
}

uint8_t* TvFFFrameReader::allocate(const uint32_t size)
{
    return _mem_manager->allocate(size);
}

void TvFFFrameReader::free(uint8_t* data)
{
    _mem_manager->free(data);
}
