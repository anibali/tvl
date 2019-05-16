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
#include "FFFRStream.h"

#include <algorithm>
using namespace std;

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
}

namespace FfFrameReader {
Stream::Stream(FormatContextPtr& formatContext, const int32_t streamID, CodecContextPtr& codecContext,
    const uint32_t bufferLength) noexcept
    : m_bufferLength(bufferLength)
    , m_formatContext(move(formatContext))
    , m_index(streamID)
    , m_codecContext(move(codecContext))
{
    // Ensure buffer length is long enough to handle the maximum number of frames a video may require
    uint32_t minFrames = getCodecDelay();
    minFrames = (m_bufferLength >= minFrames) ? m_bufferLength : minFrames;

    // Allocate ping and pong buffers
    m_bufferPing.reserve(minFrames * 2);
    m_bufferPong.reserve(minFrames * 2);

    // Set stream start time and numbers of frames
    m_startTimeStamp = getStreamStartTime();
    m_totalFrames = getStreamFrames();
    m_totalDuration = getStreamDuration();
}

Stream::FormatContextPtr::FormatContextPtr(AVFormatContext* formatContext) noexcept
    : m_formatContext(formatContext, [](AVFormatContext* p) { avformat_close_input(&p); })
{}

AVFormatContext* Stream::FormatContextPtr::operator->() const noexcept
{
    return m_formatContext.get();
}

AVFormatContext* Stream::FormatContextPtr::get() const noexcept
{
    return m_formatContext.get();
}

Stream::CodecContextPtr::CodecContextPtr(AVCodecContext* codecContext) noexcept
    : m_codecContext(codecContext, avcodec_close)
{}

AVCodecContext* Stream::CodecContextPtr::operator->() const noexcept
{
    return m_codecContext.get();
}

AVCodecContext* Stream::CodecContextPtr::get() const noexcept
{
    return m_codecContext.get();
}

uint32_t Stream::getWidth() const noexcept
{
    return m_formatContext->streams[m_index]->codecpar->width;
}

uint32_t Stream::getHeight() const noexcept
{
    return m_formatContext->streams[m_index]->codecpar->height;
}

double Stream::getAspectRatio() const noexcept
{
    if (m_formatContext->streams[m_index]->display_aspect_ratio.num) {
        return av_q2d(m_formatContext->streams[m_index]->display_aspect_ratio);
    }
    return static_cast<double>(getWidth()) / static_cast<double>(getHeight());
}

int64_t Stream::getTotalFrames() const noexcept
{
    return m_totalFrames;
}

int64_t Stream::getDuration() const noexcept
{
    return m_totalDuration;
}

double Stream::getFrameRate() const noexcept
{
    return av_q2d(m_formatContext->streams[m_index]->r_frame_rate);
}

uint32_t Stream::getFrameSize() const noexcept
{
    return av_image_get_buffer_size(static_cast<AVPixelFormat>(m_formatContext->streams[m_index]->codecpar->format),
        m_formatContext->streams[m_index]->codecpar->width, m_formatContext->streams[m_index]->codecpar->height, 0);
}

variant<bool, shared_ptr<Frame>> Stream::peekNextFrame() noexcept
{
    lock_guard<recursive_mutex> lock(m_mutex);
    // Check if we actually have any frames in the current buffer
    if (m_bufferPingHead >= m_bufferPing.size()) {
        // TODO: Async decode of next block, should start once reached the last couple of frames in a buffer
        // The swap buffer only should occur when ping buffer is exhausted and pong decode has completed
        if (!decodeNextBlock()) {
            return false;
        }
        // Swap ping and pong buffer
        swap(m_bufferPing, m_bufferPong);
        m_bufferPingHead = 0;
        // Reset the pong buffer
        m_bufferPong.resize(0);
        // Check if there are any new frames or we reached EOF
        if (m_bufferPing.size() == 0) {
            av_log(nullptr, AV_LOG_ERROR, "Cannot get a new frame, End of file has been reached.\n");
            return false;
        }
    }
    // Get frame from ping buffer
    return m_bufferPing[m_bufferPingHead];
}

variant<bool, shared_ptr<Frame>> Stream::getNextFrame() noexcept
{
    auto ret = peekNextFrame();
    if (ret.index() == 0) {
        return false;
    }
    // Remove the frame from list
    popFrame();
    return ret;
}

variant<bool, vector<shared_ptr<Frame>>> Stream::getNextFrameSequence(const vector<int64_t>& frameSequence) noexcept
{
    // Note: for best performance when using this the buffer size should be small enough to not waste to much memeory
    lock_guard<recursive_mutex> lock(m_mutex);
    vector<shared_ptr<Frame>> ret;
    int64_t start = 0;
    for (const auto& i : frameSequence) {
        if (i < start) {
            // Invalid sequence list
            av_log(nullptr, AV_LOG_ERROR,
                "Invalid sequence list passed to getNextFrameSequence(). Sequences in the list must be in ascending order.\n");
            return false;
        }
        // Remove all frames until first in sequence
        for (int64_t j = start; j < i; j++) {
            // Must peek to check there is actually a new frame
            auto err = peekNextFrame();
            if (err.index() == 0) {
                return false;
            }
            popFrame();
        }
        auto frame = getNextFrame();
        if (frame.index() == 0) {
            return false;
        }
        ret.push_back(get<1>(frame));
        start = i + 1;
    }
    return ret;
}

bool Stream::seek(const int64_t timeStamp) noexcept
{
    return seekInternal(timeStamp, false);
}

bool Stream::seekFrame(const int64_t frame) noexcept
{
    return seekFrameInternal(frame, false);
}

int64_t Stream::timeToTimeStamp(const int64_t time) const noexcept
{
    // Rescale a timestamp that is stored in microseconds (AV_TIME_BASE) to the stream timebase
    return m_startTimeStamp +
        av_rescale_q(time, av_make_q(1, AV_TIME_BASE), m_formatContext->streams[m_index]->time_base);
}

int64_t Stream::timeStampToTime(const int64_t timeStamp) const noexcept
{
    // Perform opposite operation to timeToTimeStamp
    return av_rescale_q(
        timeStamp - m_startTimeStamp, m_formatContext->streams[m_index]->time_base, av_make_q(1, AV_TIME_BASE));
}

int64_t Stream::frameToTimeStamp(const int64_t frame) const noexcept
{
    return m_startTimeStamp +
        av_rescale_q(frame, av_inv_q(m_formatContext->streams[m_index]->r_frame_rate),
            m_formatContext->streams[m_index]->time_base);
}

int64_t Stream::timeStampToFrame(const int64_t timeStamp) const noexcept
{
    return av_rescale_q(timeStamp - m_startTimeStamp, m_formatContext->streams[m_index]->r_frame_rate,
        av_inv_q(m_formatContext->streams[m_index]->time_base));
}

int64_t Stream::frameToTime(const int64_t frame) const noexcept
{
    return av_rescale_q(frame, av_make_q(AV_TIME_BASE, 1), m_formatContext->streams[m_index]->r_frame_rate);
}

int64_t Stream::timeToFrame(const int64_t time) const noexcept
{
    return av_rescale_q(time, av_make_q(1, AV_TIME_BASE), av_inv_q(m_formatContext->streams[m_index]->r_frame_rate));
}

bool Stream::decodeNextBlock() noexcept
{
    // TODO: If we are using async decode then this needs to just return if a decode is already running

    // Reset the pong buffer
    m_bufferPong.resize(0);

    // Decode the next buffer sequence
    AVPacket packet;
    av_init_packet(&packet);
    bool eof = false;
    while (true) {
        // This may or may not be a keyframe, So we just start decoding packets until we receive a valid frame
        // We do m_bufferLength packets at a time for performance even though this may result in more than
        // m_bufferLength frames being actually decoded
        auto maxPackets = m_bufferLength;
        uint32_t i = 0;
        do {
            auto ret = av_read_frame(m_formatContext.get(), &packet);
            if (ret < 0) {
                if (ret != AVERROR_EOF) {
                    char buffer[AV_ERROR_MAX_STRING_SIZE];
                    av_log(nullptr, AV_LOG_ERROR, "Failed to retrieve new frame: %s\n",
                        av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, ret));
                    return false;
                }
                eof = true;
                break;
            }

            if (m_index == packet.stream_index) {
                ++i;
                ret = avcodec_send_packet(m_codecContext.get(), &packet);
                if (ret == AVERROR(EAGAIN)) {
                    // Cannot add any more packets as must decode what we have first
                    if (!decodeNextFrames()) {
                        return false;
                    }
                    ret = avcodec_send_packet(m_codecContext.get(), &packet);
                }
                if (ret < 0) {
                    av_packet_unref(&packet);
                    char buffer[AV_ERROR_MAX_STRING_SIZE];
                    av_log(nullptr, AV_LOG_ERROR, "Failed to send packet to decoder: %s\n",
                        av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, ret));
                    return false;
                }
                // Increase the number of maxPackets if we are really close to just finishing the stream anyway
                if (i == (maxPackets - 1)) {
                    if (timeStampToFrame(packet.pts) >= (m_totalFrames - 2)) {
                        maxPackets += 2;
                    }
                }
            }
            av_packet_unref(&packet);
        } while (i < maxPackets);

        // Decode any pending frames
        if (!decodeNextFrames()) {
            return false;
        }
        // Check if we need to flush any remaining frames
        if (eof) {
            // Send flush packet to decoder
            avcodec_send_packet(m_codecContext.get(), &packet);
            if (!decodeNextFrames()) {
                return false;
            }
            // Check if we got more frames than we should have. This occurs when there are missing frames that are
            // padded in resulting in more output frames than expected.
            while (!m_bufferPong.empty()) {
                if (m_bufferPong.back()->getTimeStamp() < this->getDuration()) {
                    break;
                }
                m_bufferPong.pop_back();
            }
        }
        // Check if we have reached the buffer limit
        if ((m_bufferPong.size() >= m_bufferLength) || eof) {
            return true;
        }

        // TODO: The maximum number of frames that are needed to get a valid frame is calculated using getCodecDelay().
        // If more than that are passed without a returned frame then an error has occured.
    }
}

bool Stream::decodeNextFrames() noexcept
{
    // Loop through and retrieve all decoded frames
    Frame::FramePtr frame;
    while (true) {
        if (*frame == nullptr) {
            *frame = av_frame_alloc();
            if (*frame == nullptr) {
                av_log(nullptr, AV_LOG_ERROR, "Failed to allocate new frame\n");
                return false;
            }
        }
        const auto ret = avcodec_receive_frame(m_codecContext.get(), *frame);
        if (ret < 0) {
            av_frame_unref(*frame);
            if ((ret == AVERROR(EAGAIN)) || (ret == AVERROR_EOF)) {
                return true;
            }
            char buffer[AV_ERROR_MAX_STRING_SIZE];
            av_log(nullptr, AV_LOG_ERROR, "Failed to receive decoded frame: %s\n",
                av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, ret));
            return false;
        }

        // Calculate time stamp for frame
        const auto timeStamp = timeStampToTime(frame->best_effort_timestamp);
        const auto frameNum = timeStampToFrame(frame->best_effort_timestamp);

        // Check if we have skipped a frame
        if (!m_bufferPong.empty()) {
            // TODO: Handle case where pong is empty but a frame was still skipped between now and last entry in ping
            // (which has already been popped by the time this function is called)
            const auto previous = m_bufferPong.back();
            if (frameNum != previous->getFrameNumber() + 1) {
                // Fill in missing frames by duplicating the old one
                auto fillFrameNum = previous->getFrameNumber();
                int64_t fillTimeStamp;
                for (auto i = previous->getFrameNumber() + 1; i < frameNum; i++) {
                    ++fillFrameNum;
                    fillTimeStamp = frameToTime(fillFrameNum);
                    Frame::FramePtr frameClone(av_frame_clone(*frame));
                    m_bufferPong.emplace_back(make_shared<Frame>(frameClone, fillTimeStamp, fillFrameNum));
                }
            }
        }

        // TODO: Need to determine type of memory pointer requested and perform a memory move to the required memory
        // space
        /*
        if (m_frame->format == hw_pix_fmt) {
            AVFrame *sw_frame = av_frame_alloc()
            if ((ret = av_hwframe_transfer_data(sw_frame, m_frame, 0)) < 0) {
                error
            }
            tmp_frame = sw_frame;
        }
        else { tmp_frame = frame; }
        */
        // TODO: Need to convert to proper colour space format

        // Add the new frame to the pong buffer
        m_bufferPong.emplace_back(make_shared<Frame>(frame, timeStamp, frameNum));
    }
}

void Stream::popFrame() noexcept
{
    if (m_bufferPingHead >= m_bufferPing.size()) {
        av_log(nullptr, AV_LOG_ERROR, "No more frames to pop\n");
        return;
    }
    // Release reference and pop frame
    m_bufferPing[m_bufferPingHead++] = make_shared<Frame>();
}

bool Stream::seekInternal(const int64_t timeStamp, const bool recursed) noexcept
{
    lock_guard<recursive_mutex> lock(m_mutex);
    // Check if we actually have any frames in the current buffer
    if (m_bufferPing.size() > 0) {
        // Check if the frame is in the current buffer
        if ((m_bufferPingHead < m_bufferPing.size()) && (timeStamp >= m_bufferPing[m_bufferPingHead]->getTimeStamp()) &&
            (timeStamp <= m_bufferPing.back()->getTimeStamp())) {
            // Dump all frames before requested one
            while (true) {
                // Get next frame
                auto ret = peekNextFrame();
                if (ret.index() == 0) {
                    return false;
                }
                // Check if we have found our requested time stamp
                const auto frame = get<1>(ret);
                if (timeStamp <= frame->getTimeStamp()) {
                    break;
                }
                // Check if the timestamp does not exactly match but is within the timestamp range of the next frame
                if ((timeStamp > frame->getTimeStamp()) && (timeStamp < (frame->getTimeStamp() + frameToTime(1)))) {
                    break;
                }
                // Remove frames from ping buffer
                popFrame();
            }
            return true;
        }

        // Check if this is a forward seek within some predefined small range. If so then just continue reading
        // packets from the current position into buffer.
        if (timeStamp > m_bufferPing.back()->getTimeStamp()) {
            // Forward decode if within some predefined range of existing point. If this is a recurse then we need to
            // compensate for potentially huge gaps between seek frames.
            const int64_t forwardRange = (!recursed) ? m_bufferLength * 3 : 1000;
            const auto timeRange = frameToTime(forwardRange);
            if (timeStamp <= m_bufferPing.back()->getTimeStamp() + timeRange) {
                // Loop through until the requested timestamp is found (or nearest timestamp rounded up if exact match
                // could not be found). Discard all frames occuring before timestamp

                // Clean out current buffer
                m_bufferPing.resize(0);
                m_bufferPingHead = 0;

                // Decode the next block of frames
                if (peekNextFrame().index() == 0) {
                    return false;
                }

                // Search through buffer until time stamp is found
                return seekInternal(timeStamp, true);
            }
        }
    }

    // If we have recursed and still havnt found the frame then we never will
    if (recursed) {
        av_log(nullptr, AV_LOG_ERROR, "Failed to seek to specified time stamp %" PRId64 "\n", timeStamp);
        return false;
    }

    // Seek to desired timestamp
    avcodec_flush_buffers(m_codecContext.get());
    const auto localTimeStamp = timeToTimeStamp(timeStamp) + m_startTimeStamp;
    const auto err = avformat_seek_file(m_formatContext.get(), m_index, INT64_MIN, localTimeStamp, localTimeStamp, 0);
    if (err < 0) {
        char buffer[AV_ERROR_MAX_STRING_SIZE];
        av_log(nullptr, AV_LOG_ERROR, "Failed seeking to specified time stamp %" PRId64 ": %s\n", timeStamp,
            av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, err));
        return false;
    }

    // Clean out current buffer
    m_bufferPing.resize(0);
    m_bufferPingHead = 0;

    // Decode the next block of frames
    if (peekNextFrame().index() == 0) {
        return false;
    }

    // Search through buffer until time stamp is found
    return seekInternal(timeStamp, true);
}

bool Stream::seekFrameInternal(const int64_t frame, const bool recursed) noexcept
{
    lock_guard<recursive_mutex> lock(m_mutex);
    // Check if we actually have any frames in the current buffer
    if (m_bufferPing.size() > 0) {
        // Check if the frame is in the current buffer
        if ((m_bufferPingHead < m_bufferPing.size()) && (frame >= m_bufferPing[m_bufferPingHead]->getFrameNumber()) &&
            (frame <= m_bufferPing.back()->getFrameNumber())) {
            // Dump all frames before requested one
            while (true) {
                // Get next frame
                auto ret = peekNextFrame();
                if (ret.index() == 0) {
                    return false;
                }
                // Check if we have found our requested frame
                if (frame <= get<1>(ret)->getFrameNumber()) {
                    break;
                }
                // Remove frames from ping buffer
                popFrame();
            }
            return true;
        }

        // Check if this is a forward seek within some predefined small range. If so then just continue reading
        // packets from the current position into buffer.
        if (frame > m_bufferPing.back()->getFrameNumber()) {
            // Forward decode if within some predefined range of existing point. If this is a recurse then we need to
            // compensate for potentially huge gaps between seek frames.
            const int64_t frameRange = (!recursed) ? m_bufferLength * 3 : 1000;
            if (frame <= m_bufferPing.back()->getFrameNumber() + frameRange) {
                while (true) {
                    auto ret = peekNextFrame();
                    if (ret.index() == 0) {
                        return false;
                    }
                    // Check if we have found our requested time stamp
                    if (frame <= get<1>(ret)->getFrameNumber()) {
                        break;
                    }
                    // Remove frames from ping buffer
                    popFrame();
                }
                return true;
            }
        }
    }

    // If we have recursed and still havnt found the frame then we never will
    if (recursed || !m_frameSeekSupported) {
        if (m_frameSeekSupported) {
            m_frameSeekSupported = false;
            av_log(nullptr, AV_LOG_ERROR,
                "Failed to seek to specified frame %" PRId64 " (retrying using timestamp based seek)\n", frame);
        } else if (recursed) {
            return false;
        }

        // Try and seek just using a timestamp
        return seek(frameToTime(frame));
    }

    // Seek to desired timestamp
    avcodec_flush_buffers(m_codecContext.get());
    const auto frameInternal = frame + timeStampToFrame(m_startTimeStamp);
    const auto err =
        avformat_seek_file(m_formatContext.get(), m_index, INT64_MIN, frameInternal, frameInternal, AVSEEK_FLAG_FRAME);
    if (err < 0) {
        m_frameSeekSupported = false;
        char buffer[AV_ERROR_MAX_STRING_SIZE];
        av_log(nullptr, AV_LOG_ERROR,
            "Failed to seek to specified frame %" PRId64 ": %s (retrying using timestamp based seek)\n", frame,
            av_make_error_string(buffer, AV_ERROR_MAX_STRING_SIZE, err));

        // Try and seek just using a timestamp
        return seek(frameToTime(frame));
    }

    // Clean out current buffer
    m_bufferPing.resize(0);
    m_bufferPingHead = 0;

    // Decode the next block of frames
    if (peekNextFrame().index() == 0) {
        return false;
    }

    // Search through buffer until time stamp is found
    return seekFrameInternal(frame, true);
}

int32_t Stream::getCodecDelay() const noexcept
{
    return std::max(((m_codecContext->codec->capabilities & AV_CODEC_CAP_DELAY) ? m_codecContext->delay : 0) +
            m_codecContext->has_b_frames,
        1);
}

int64_t Stream::getStreamStartTime() const noexcept
{
    // First check if the stream has a start timeStamp
    AVStream* stream = m_formatContext->streams[m_index];
    if (stream->start_time != int64_t(AV_NOPTS_VALUE)) {
        return stream->start_time;
    }
    // Seek to the first frame in the video to get information directly from it
    avcodec_flush_buffers(m_codecContext.get());
    int64_t startDts = 0LL;
    if (stream->first_dts != int64_t(AV_NOPTS_VALUE)) {
        startDts = std::min(startDts, stream->first_dts);
    }
    if (av_seek_frame(m_formatContext.get(), m_index, startDts, AVSEEK_FLAG_BACKWARD) < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Failed to determine stream start time\n");
        return 0;
    }
    AVPacket packet;
    av_init_packet(&packet);
    // Read frames until we get one for the video stream that contains a valid PTS or DTS.
    auto startTimeStamp = int64_t(AV_NOPTS_VALUE);
    const auto maxPackets = getCodecDelay();
    // Loop through multiple packets to take into account b-frame reordering issues
    for (int32_t i = 0; i < maxPackets;) {
        if (av_read_frame(m_formatContext.get(), &packet) < 0) {
            return 0;
        }
        if (packet.stream_index == m_index) {
            // Get the Presentation time stamp for the packet, if this value is not set then try the Decompression time
            // stamp
            auto pts = packet.pts;
            if (pts == int64_t(AV_NOPTS_VALUE)) {
                pts = packet.dts;
            }
            if ((pts != int64_t(AV_NOPTS_VALUE)) &&
                ((pts < startTimeStamp) || (startTimeStamp == int64_t(AV_NOPTS_VALUE)))) {
                startTimeStamp = pts;
            }
            ++i;
        }
        av_packet_unref(&packet);
    }
    // Seek back to start of file so future reads continue back at start
    av_seek_frame(m_formatContext.get(), m_index, startDts, AVSEEK_FLAG_BACKWARD);
    return (startTimeStamp != int64_t(AV_NOPTS_VALUE)) ? startTimeStamp : 0;
}

int64_t Stream::getStreamFrames() const noexcept
{
    AVStream* stream = m_formatContext->streams[m_index];
    // Check if the number of frames is specified in the stream
    if (stream->nb_frames > 0) {
        return stream->nb_frames - timeStampToFrame(m_startTimeStamp * 2);
    }

    // Attempt to calculate from stream duration, time base and fps
    if (stream->duration > 0) {
        return timeStampToFrame(int64_t(stream->duration));
    }

    // If we are at this point then the only option is to scan the entire file and check the DTS/PTS.
    int64_t foundTimeStamp = m_startTimeStamp;

    // Seek last key-frame.
    avcodec_flush_buffers(m_codecContext.get());
    const auto maxSeek = frameToTimeStamp(1UL << 29UL);
    if (avformat_seek_file(m_formatContext.get(), m_index, INT64_MIN, maxSeek, maxSeek, 0) < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Failed to determine number of frames in stream\n");
        return 0;
    }

    // Read up to last frame, extending max PTS for every valid PTS value found for the video stream.
    AVPacket packet;
    av_init_packet(&packet);
    while (av_read_frame(m_formatContext.get(), &packet) >= 0) {
        if (packet.stream_index == m_index) {
            auto found = packet.pts;
            if (found == int64_t(AV_NOPTS_VALUE)) {
                found = packet.dts;
            }
            if (found > foundTimeStamp) {
                foundTimeStamp = found;
            }
        }
        av_packet_unref(&packet);
    }

    // Seek back to start of file so future reads continue back at start
    int64_t startDts = 0LL;
    if (stream->first_dts != int64_t(AV_NOPTS_VALUE)) {
        startDts = std::min(startDts, stream->first_dts);
    }
    av_seek_frame(m_formatContext.get(), m_index, startDts, AVSEEK_FLAG_BACKWARD);

    // The detected value is the index of the last frame plus one
    return timeStampToFrame(foundTimeStamp) + 1;
}

int64_t Stream::getStreamDuration() const noexcept
{
    // First try and get the format duration if specified. For some formats this durations can override the duration
    // specified within each stream which is why it should be checked first.
    AVStream* stream = m_formatContext->streams[m_index];
    if (m_formatContext->duration > 0) {
        return m_formatContext->duration -
            timeStampToTime(m_startTimeStamp * 2); //*2 To avoid the minus in timeStampToTime
    }

    // Check if the duration is specified in the stream
    if (stream->duration > 0) {
        return timeStampToTime(stream->duration);
    }

    // If we are at this point then the only option is to scan the entire file and check the DTS/PTS.
    int64_t foundTimeStamp = m_startTimeStamp;

    // Seek last key-frame.
    avcodec_flush_buffers(m_codecContext.get());
    const auto maxSeek = frameToTimeStamp(1UL << 29UL);
    if (avformat_seek_file(m_formatContext.get(), m_index, INT64_MIN, maxSeek, maxSeek, 0) < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Failed to determine stream duration\n");
        return 0;
    }

    // Read up to last frame, extending max PTS for every valid PTS value found for the video stream.
    AVPacket packet;
    av_init_packet(&packet);
    while (av_read_frame(m_formatContext.get(), &packet) >= 0) {
        if (packet.stream_index == m_index) {
            auto found = packet.pts;
            if (found == int64_t(AV_NOPTS_VALUE)) {
                found = packet.dts;
            }
            if (found > foundTimeStamp) {
                foundTimeStamp = found;
            }
        }
        av_packet_unref(&packet);
    }

    // Seek back to start of file so future reads continue back at start
    int64_t startDts = 0LL;
    if (stream->first_dts != int64_t(AV_NOPTS_VALUE)) {
        startDts = std::min(startDts, stream->first_dts);
    }
    av_seek_frame(m_formatContext.get(), m_index, startDts, AVSEEK_FLAG_BACKWARD);

    // The detected value is timestamp of the last detected packet plus the duration of that frame
    return timeStampToTime(foundTimeStamp) + frameToTime(1);
}
} // namespace FfFrameReader