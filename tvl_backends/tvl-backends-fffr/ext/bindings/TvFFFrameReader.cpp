#include "TvFFFrameReader.h"

#include <cstring>
#include <memory>
#include <stdexcept>

std::map<int, std::shared_ptr<std::remove_pointer<CUcontext>::type>> TvFFFrameReader::_contexts;

bool TvFFFrameReader::init_context(const int gpu_index)
{
    // Detect shared context with runtime api
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    CUcontext context;
    CUdevice dev;
    if (cuDeviceGet(&dev, gpu_index) != CUDA_SUCCESS) {
        return false;
    }
    if (cuDevicePrimaryCtxRetain(&context, dev) != CUDA_SUCCESS) {
        return false;
    }
    _contexts[gpu_index] = std::shared_ptr<std::remove_pointer<CUcontext>::type>(
        context, [dev](CUcontext) { cuDevicePrimaryCtxRelease(dev); });
    return true;
}

TvFFFrameReader::TvFFFrameReader(ImageAllocator* image_allocator, const std::string& filename, const int gpu_index,
    const int out_width, const int out_height)
    : _image_allocator(image_allocator)
    , _filename(filename)
{
    // Quiet the log
    Ffr::setLogLevel(Ffr::LogLevel::Quiet);

    switch (image_allocator->get_data_type()) {
        case ImageAllocator::UINT8: {
            _pixel_format = Ffr::PixelFormat::RGB8P;
            break;
        }
        case ImageAllocator::FLOAT32: {
            _pixel_format = Ffr::PixelFormat::RGB32FP;
            break;
        }
        default: {
            throw std::runtime_error("Unsupported data type.");
        }
    }

    // Set up decoding options
    Ffr::DecoderOptions options;
    if (gpu_index >= 0) {
        if (_contexts.find(gpu_index) == _contexts.end()) {
            if (!init_context(gpu_index)) {
                throw std::runtime_error("CUDA context creation failed.");
            }
        }

        options.m_type = Ffr::DecodeType::Cuda;
        options.m_outputHost = false;
        options.m_format = Ffr::PixelFormat::Auto; // Keep pixel format the same
        options.m_context = _contexts[gpu_index].get();
    } else {
        // Use inbuilt software conversion
        options.m_format = _pixel_format;
    }
    options.m_scale.m_width = out_width;
    options.m_scale.m_height = out_height;

    // Create a decoding stream
    _stream = Ffr::Stream::getStream(filename, options);
    if (_stream == nullptr) {
        throw std::runtime_error("Stream creation failed.");
    }
}

std::string TvFFFrameReader::get_filename() const
{
    return _filename;
}

int TvFFFrameReader::get_width() const
{
    return _stream->getWidth();
}

int TvFFFrameReader::get_height() const
{
    return _stream->getHeight();
}

double TvFFFrameReader::get_duration() const
{
    return static_cast<double>(_stream->getDuration()) / 1000000.0;
}

double TvFFFrameReader::get_frame_rate() const
{
    return _stream->getFrameRate();
}

int64_t TvFFFrameReader::get_number_of_frames() const
{
    return _stream->getTotalFrames();
}

void TvFFFrameReader::seek(const float time_secs)
{
    const auto time = static_cast<int64_t>(time_secs * 1000000.0f);
    const bool ret = _stream->seek(time);
    if (!ret) {
        throw std::runtime_error("Seek failed.");
    }
}

uint8_t* TvFFFrameReader::convert_frame(std::shared_ptr<Ffr::Frame> frame)
{
    // Check if known pixel format
    if (frame->getPixelFormat() == Ffr::PixelFormat::Auto) {
        throw std::runtime_error("Unknown pixel format.");
    }

    // Get frame dimensions
    const int width = frame->getWidth();
    const int height = frame->getHeight();
    const int lineSize = Ffr::getImageLineStep(_pixel_format, width, 0);

    // Allocate new memory to store frame data
    const auto newData = reinterpret_cast<uint8_t*>(_image_allocator->allocate_frame(width, height, lineSize, 32));
    if (newData == nullptr) {
        throw std::runtime_error("Memory allocation for frame image failed.");
    }

    // Copy/Convert image data into output
    if (frame->getDataType() == Ffr::DecodeType::Cuda) {
        if (!Ffr::convertFormat(frame, newData, _pixel_format)) {
            _image_allocator->free_frame(newData);
            throw std::runtime_error("Pixel format conversion failed.");
        }
    } else {
        auto plane = newData;
        for (int32_t i = 0; i < 3; i++) {
            const auto planeData = frame->getFrameData(i);
            const auto size = planeData.second * frame->getHeight();
            std::memcpy(plane, planeData.first, size);
            plane += size;
        }
    }

    return newData;
}

uint8_t* TvFFFrameReader::read_frame()
{
    // Get next frame
    const auto frame = _stream->getNextFrame();
    if (frame == nullptr) {
        if (_stream->isEndOfFile()) {
            // This is an EOF error
            return nullptr;
        }
        throw std::runtime_error("Failed to get the next frame.");
    }
    return convert_frame(frame);
}

int64_t TvFFFrameReader::read_frame_sequence(int64_t* offsets, int n_frames, uint8_t** frames)
{
    // Read a sequence of frames. We request all of them, but we won't necessarily be able
    // to get all of them at once.
    const std::vector<int64_t> frame_sequence(offsets, offsets + n_frames);
    const auto frames_vector = _stream->getNextFrameSequence(frame_sequence);
    uint32_t index = 0;
    for (auto& i : frames_vector) {
        frames[index] = convert_frame(i);
        ++index;
    }
    if (frames_vector.size() != static_cast<size_t>(n_frames)) {
        if (_stream->isEndOfFile()) {
            // Return false to indicate "end of file".
            return false;
        }
        throw std::runtime_error("Failed to get the next frame sequence.");
    }
    return frames_vector.size();
}