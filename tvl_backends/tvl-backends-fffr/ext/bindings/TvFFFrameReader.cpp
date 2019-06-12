#include "TvFFFrameReader.h"

#include <libavutil/avutil.h>
#include <memory>
#include <stdexcept>

TvFFFrameReader::TvFFFrameReader(ImageAllocator* image_allocator, const std::string& filename,
    const int gpu_index, const int out_width, const int out_height)
    : _image_allocator(image_allocator)
    , _filename(filename)
{
    // Quiet the log
    Ffr::setLogLevel(Ffr::LogLevel::Quiet);

    switch (image_allocator->get_data_type()) {
        case ImageAllocator::UINT8: {
            _pixel_format = Ffr::PixelFormat::GBR8P;
            break;
        }
        case ImageAllocator::FLOAT32: {
            _pixel_format = Ffr::PixelFormat::GBR32FP;
            break;
        }
        default: {
            throw std::runtime_error("Unsupported data type.");
        }
    }

    // Set up decoding options
    Ffr::DecoderOptions options;
    if (gpu_index >= 0) {
        options.m_type = Ffr::DecodeType::Cuda;
        options.m_device = gpu_index;
        options.m_outputHost = false;
        options.m_format = Ffr::PixelFormat::Auto; // Keep pixel format the same
    } else {
        // Use inbuilt software conversion
        options.m_format = _pixel_format;
    }
    options.m_scale.m_width = out_width;
    options.m_scale.m_height = out_height;

    // Create a decoding stream
    const auto ret = Ffr::Stream::getStream(filename, options);
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
    return static_cast<double>(_stream->getDuration()) / static_cast<double>(AV_TIME_BASE);
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

uint8_t* TvFFFrameReader::read_frame()
{
    // Get next frame
    const auto ret = _stream->getNextFrame();
    if (ret.index() == 0) {
        if (std::get<0>(ret) == true) {
            //This is an EOF error
            return nullptr;
        }
        throw std::runtime_error("Failed to get the next frame.");
    }
    const auto frame = std::get<1>(ret);

    // Check if known pixel format
    if (frame->getPixelFormat() == Ffr::PixelFormat::Auto) {
        throw std::runtime_error("Unknown pixel format.");
    }

    // Get frame dimensions
    const int width = frame->getWidth();
    const int height = frame->getHeight();
    const int lineSize = frame->getFrameData(0).second;

    // Allocate new memory to store frame data
    const auto outFrameSize = Ffr::getImageSize(_pixel_format, width, height);
    const auto newData = reinterpret_cast<uint8_t*>(
        _image_allocator->allocate_frame(width, height, lineSize));
    if (newData == nullptr) {
        throw std::runtime_error("Memory allocation for frame image failed.");
    }

    // Calculate memory locations for each plane
    void* outPlanes[3];
    size_t ignored;
    outPlanes[0] = newData;
    std::align(32, outFrameSize, outPlanes[0], ignored);
    for (int32_t i = 1; i < 3; i++) {
        const auto planeData = frame->getFrameData(i);
        const uint32_t planeSize = planeData.second * frame->getHeight();
        outPlanes[i] = reinterpret_cast<uint8_t*>(outPlanes[i - 1]) + planeSize;
        std::align(32, outFrameSize, outPlanes[i], ignored);
    }
    // Swap output planes to match GBR ordering
    const auto backup = outPlanes[0];
    outPlanes[0] = outPlanes[1];
    outPlanes[1] = outPlanes[2];
    outPlanes[2] = backup;

    // Copy/Convert image data into output
    if (frame->getDataType() == Ffr::DecodeType::Cuda) {
        if (!Ffr::convertFormat(frame, reinterpret_cast<uint8_t**>(outPlanes), _pixel_format)) {
            _image_allocator->free_frame(newData);
            throw std::runtime_error("Pixel format conversion failed.");
        }
    } else {
        for (int32_t i = 0; i < 3; i++) {
            const auto planeData = frame->getFrameData(i);
            memcpy(outPlanes[i], planeData.first, planeData.second * frame->getHeight());
        }
    }
    return newData;
}
