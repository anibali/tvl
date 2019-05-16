/**
 * Copyright 2019 Matthew Oliver
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "FFFRDecoderContext.h"

#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
}

using namespace std;

namespace FfFrameReader {
const DecoderContext::DeviceContextPtr& getDeviceContext(DecoderContext* context) noexcept
{
    return context->m_deviceContext;
}

const optional<DecoderContext::DecoderOptions::Allocator>& getAllocator(DecoderContext* context) noexcept
{
    return context->m_allocator;
}

static void poolFree(void* opaque, uint8_t* data) noexcept
{
    const auto decoder = reinterpret_cast<DecoderContext*>(opaque);
    // Call the custom free function
    try {
        getAllocator(decoder).value().m_free(data);
    } catch (...) {
    }
}

static AVBufferRef* poolAllocator(void* opaque, const int32_t size) noexcept
{
    const auto decoder = reinterpret_cast<DecoderContext*>(opaque);
    // Call the custom allocator
    uint8_t* data;
    try {
        data = getAllocator(decoder).value().m_allocate(size);
    } catch (...) {
        return nullptr;
    }

    const auto ret = av_buffer_create(data, size, poolFree, opaque, 0);
    if (!ret) {
        av_log(nullptr, AV_LOG_ERROR, "Failed to allocate frame using custom allocator\n");
        return nullptr;
    }
    return ret;
}

static enum AVPixelFormat getHardwareFormatNvdec(AVCodecContext* context, const enum AVPixelFormat* pixelFormats)
{
    enum AVPixelFormat pixelFormat;
    for (int i = 0;; i++) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(context->codec, i);
        if (!config) {
            av_log(nullptr, AV_LOG_ERROR, "Decoder does not support device type: %s, %s\n", context->codec->name,
                av_hwdevice_get_type_name(AV_HWDEVICE_TYPE_CUDA));
            return AV_PIX_FMT_NONE;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == AV_HWDEVICE_TYPE_CUDA) {
            pixelFormat = config->pix_fmt;
            break;
        }
    }
    for (const enum AVPixelFormat* p = pixelFormats; *p != -1; p++) {
        if (*p == pixelFormat) {
            if (context->opaque != nullptr) {
                const auto opaque = reinterpret_cast<DecoderContext*>(context->opaque);
                const auto& deviceContext = getDeviceContext(opaque);
                // Create a custom frame pool
                context->hw_frames_ctx = av_hwframe_ctx_alloc(deviceContext.get());
                if (!context->hw_frames_ctx) {
                    av_log(nullptr, AV_LOG_ERROR, "Failed to allocate custom hardware frame context\n");
                    return AV_PIX_FMT_NONE;
                }
                auto framesContext = reinterpret_cast<AVHWFramesContext*>(context->hw_frames_ctx->data);

                framesContext->format = AV_PIX_FMT_CUDA;
                framesContext->sw_format = context->sw_pix_fmt;
                framesContext->width = FFALIGN(context->coded_width, 2);
                framesContext->height = FFALIGN(context->coded_height, 2);
                framesContext->initial_pool_size = 32;

                const auto size = av_image_get_buffer_size(
                    framesContext->sw_format, framesContext->width, framesContext->height, 256);
                if (size < 0) {
                    av_buffer_unref(&context->hw_frames_ctx);
                    char buffer[AV_ERROR_MAX_STRING_SIZE];
                    av_log(nullptr, AV_LOG_ERROR, "Failed to get image size: %s\n",
                        av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, size));
                    return AV_PIX_FMT_NONE;
                }

                framesContext->pool = av_buffer_pool_init2(size, opaque, poolAllocator, nullptr);
                if (framesContext->pool == nullptr) {
                    av_buffer_unref(&context->hw_frames_ctx);
                    av_log(nullptr, AV_LOG_ERROR, "Failed to allocate buffer pool\n");
                    return AV_PIX_FMT_NONE;
                }

                const auto err = av_hwframe_ctx_init(context->hw_frames_ctx);
                if (err < 0) {
                    av_buffer_unref(&context->hw_frames_ctx);
                    char buffer[AV_ERROR_MAX_STRING_SIZE];
                    av_log(nullptr, AV_LOG_ERROR, "Failed to initialise custom hardware frame context: %s\n",
                        av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, err));
                    return AV_PIX_FMT_NONE;
                }
            }

            return *p;
        }
    }
    av_log(nullptr, AV_LOG_ERROR, "Failed to get hardware surface format\n");
    return AV_PIX_FMT_NONE;
}

static enum AVHWDeviceType decodeTypeToFFmpeg(const DecoderContext::DecodeType type)
{
    if (type == DecoderContext::DecodeType::Nvdec) {
        return AV_HWDEVICE_TYPE_CUDA;
    }
    return AV_HWDEVICE_TYPE_NONE;
}

DecoderContext::DecoderOptions::DecoderOptions(const DecodeType type) noexcept
    : m_type(type)
{}

bool DecoderContext::DecoderOptions::operator==(const DecoderOptions& other) const noexcept
{
    const bool ret = this->m_type == other.m_type && this->m_bufferLength == other.m_bufferLength &&
        this->m_device == other.m_device;
    if (!ret) {
        return ret;
    }
    if (this->m_type == DecodeType::Software) {
        return true;
    }
    try {
        if (this->m_type == DecodeType::Nvdec) {
            return any_cast<CUcontext>(this->m_context) == any_cast<CUcontext>(other.m_context);
        }
    } catch (...) {
        return false;
    }
    return false;
}

bool DecoderContext::DecoderOptions::operator!=(const DecoderOptions& other) const noexcept
{
    return !(*this == other);
}

bool DecoderContext::DecoderOptions::operator<(const DecoderOptions& other) const noexcept
{
    if (this->m_type < other.m_type) {
        return true;
    }
    if (other.m_type < this->m_type) {
        return false;
    }
    if (this->m_bufferLength < other.m_bufferLength) {
        return true;
    }
    if (other.m_bufferLength < this->m_bufferLength) {
        return false;
    }
    try {
        if (this->m_type == DecodeType::Nvdec) {
            if (any_cast<CUcontext>(this->m_context) < any_cast<CUcontext>(other.m_context)) {
                return true;
            }
            if (any_cast<CUcontext>(other.m_context) < any_cast<CUcontext>(this->m_context)) {
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return this->m_device < other.m_device;
}

DecoderContext::DeviceContextPtr::DeviceContextPtr(AVBufferRef* deviceContext) noexcept
    : m_deviceContext(deviceContext, [](AVBufferRef* p) { av_buffer_unref(&p); })
{}

AVBufferRef* DecoderContext::DeviceContextPtr::operator->() const noexcept
{
    return m_deviceContext.get();
}

AVBufferRef* DecoderContext::DeviceContextPtr::get() const noexcept
{
    return m_deviceContext.get();
}

DecoderContext::DecoderContext(const DecoderOptions& options) noexcept
    : m_bufferLength(options.m_bufferLength)
{
    // TODO: Allow specifying a filter chain of color conversion, scale and cropping
    if (options.m_type != DecodeType::Software) {
        // Create device specific options
        string device;
        try {
            device = to_string(options.m_device);
        } catch (...) {
        }
        const auto type = decodeTypeToFFmpeg(options.m_type);

        // Check if we should create a custom hwdevice
        if (options.m_context.has_value()) {
            DeviceContextPtr tempDevice(av_hwdevice_ctx_alloc(type));
            if (tempDevice.get() == nullptr) {
                av_log(nullptr, AV_LOG_ERROR, "Failed to create custom hardware device\n");
                return;
            }
            auto* deviceContext = reinterpret_cast<AVHWDeviceContext*>(tempDevice->data);
            deviceContext->free = nullptr;
            if (type == AV_HWDEVICE_TYPE_CUDA) {
                if (options.m_context.type() != typeid(CUcontext)) {
                    av_log(nullptr, AV_LOG_ERROR, "Specified device context does not match the required type\n");
                    return;
                }
                auto* cudaDevice = reinterpret_cast<AVCUDADeviceContext*>(deviceContext->hwctx);
                try {
                    cudaDevice->cuda_ctx = any_cast<CUcontext>(options.m_context);
                } catch (...) {
                }
            }
            const auto ret = av_hwdevice_ctx_init(tempDevice.get());
            if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Failed to init custom hardware device\n");
                return;
            }
            // Move the temp device to the internal one
            m_deviceContext = move(tempDevice);
        } else {
            // Create the hardware context for decoding
            AVBufferRef* deviceContext;
            const int err = av_hwdevice_ctx_create(&deviceContext, type, device.c_str(), nullptr, 0);
            if (err < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Failed to create specified hardware device\n");
                return;
            }
            m_deviceContext = DeviceContextPtr(deviceContext);
        }
        // Set internal values now we have successfully created a hardware device (otherwise stays at default)
        m_deviceType = options.m_type;
        // Set internal allocator
        m_allocator = options.m_allocator;
    }
}

variant<bool, shared_ptr<Stream>> DecoderContext::getStream(const string& filename) const noexcept
{
    AVFormatContext* formatPtr = nullptr;
    auto ret = avformat_open_input(&formatPtr, filename.c_str(), nullptr, nullptr);
    Stream::FormatContextPtr tempFormat(formatPtr);
    if (ret < 0) {
        char buffer[AV_ERROR_MAX_STRING_SIZE];
        av_log(nullptr, AV_LOG_ERROR, "Failed to open input stream '%s': %s\n", filename.c_str(),
            av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, ret));
        return false;
    }
    ret = avformat_find_stream_info(tempFormat.get(), nullptr);
    if (ret < 0) {
        char buffer[AV_ERROR_MAX_STRING_SIZE];
        av_log(nullptr, AV_LOG_ERROR, "Failed finding stream information '%s': %s\n", filename.c_str(),
            av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, ret));
        return false;
    }

    // Get the primary video stream
    AVCodec* decoder = nullptr;
    ret = av_find_best_stream(tempFormat.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (ret < 0) {
        char buffer[AV_ERROR_MAX_STRING_SIZE];
        av_log(nullptr, AV_LOG_ERROR, "Failed to find video stream in file '%s': %s\n", filename.c_str(),
            av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, ret));
        return false;
    }
    AVStream* stream = tempFormat->streams[ret];
    int32_t index = ret;

    if (m_deviceType != DecodeType::Software) {
        // Check if required codec is supported
        for (int i = 0;; i++) {
            const AVCodecHWConfig* config = avcodec_get_hw_config(decoder, i);
            if (config == nullptr) {
                av_log(nullptr, AV_LOG_ERROR, "Decoder does not support device type: %s, %s\n", decoder->name,
                    av_hwdevice_get_type_name(decodeTypeToFFmpeg(m_deviceType)));
                return false;
            }
            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
                config->device_type == AV_HWDEVICE_TYPE_CUDA) {
                break;
            }
        }
    }

    // Create a decoder context
    Stream::CodecContextPtr tempCodec(avcodec_alloc_context3(decoder));
    if (tempCodec.get() == nullptr) {
        av_log(nullptr, AV_LOG_ERROR, "Failed allocating decoder context %s\n", filename.c_str());
        return false;
    }

    ret = avcodec_parameters_to_context(tempCodec.get(), stream->codecpar);
    if (ret < 0) {
        char buffer[AV_ERROR_MAX_STRING_SIZE];
        av_log(nullptr, AV_LOG_ERROR, "Failed copying parameters to decoder context '%s': %s\n", filename.c_str(),
            av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, ret));
        return false;
    }

    // Setup any required hardware decoding parameters
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "refcounted_frames", "1", 0);
    if (m_deviceType != DecodeType::Software) {
        if (m_deviceType == DecodeType::Nvdec) {
            tempCodec->get_format = getHardwareFormatNvdec;
            if (m_allocator.has_value()) {
                tempCodec->opaque = const_cast<void*>(reinterpret_cast<const void*>(this));
            } else {
                tempCodec->opaque = nullptr;
            }
        } else {
            av_log(nullptr, AV_LOG_ERROR, "Hardware Device not properly implemented\n");
            return false;
        }
        tempCodec->hw_device_ctx = av_buffer_ref(m_deviceContext.get());
        // Enable extra hardware frames to ensure we don't run out of buffers
        const string buffers = to_string(m_bufferLength);
        av_dict_set(&opts, "extra_hw_frames", buffers.c_str(), 0);
    } else {
        av_dict_set(&opts, "threads", "auto", 0);
    }
    ret = avcodec_open2(tempCodec.get(), decoder, &opts);
    if (ret < 0) {
        char buffer[AV_ERROR_MAX_STRING_SIZE];
        av_log(nullptr, AV_LOG_ERROR, "Failed opening decoder for %s: %s\n", filename.c_str(),
            av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, ret));
        return false;
    }

    // Make a new stream object and add it to internal list
    return make_shared<Stream>(tempFormat, index, tempCodec, m_bufferLength);
}
} // namespace FfFrameReader