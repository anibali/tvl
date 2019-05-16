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
#pragma once
#include "FFFRFrame.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

struct AVFormatContext;
struct AVCodecContext;

namespace FfFrameReader {
class Stream
{
    friend class DecoderContext;

private:
    class FormatContextPtr
    {
        friend class DecoderContext;
        friend class Stream;

    private:
        explicit FormatContextPtr(AVFormatContext* formatContext) noexcept;

        [[nodiscard]] AVFormatContext* get() const noexcept;

        AVFormatContext* operator->() const noexcept;

        std::shared_ptr<AVFormatContext> m_formatContext = nullptr;
    };

    class CodecContextPtr
    {
        friend class DecoderContext;
        friend class Stream;

    private:
        explicit CodecContextPtr(AVCodecContext* codecContext) noexcept;

        [[nodiscard]] AVCodecContext* get() const noexcept;

        AVCodecContext* operator->() const noexcept;

        std::shared_ptr<AVCodecContext> m_codecContext = nullptr;
    };

public:
    Stream() = delete;

    /**
     * Constructor.
     * @param [in,out] formatContext Context for the format. This is reset to nullptr on function exit.
     * @param          streamID      Index of the stream.
     * @param [in,out] codecContext  Context for the codec. This is reset to nullptr on function exit.
     * @param          bufferLength  Length of the internal decode buffer.
     */
    Stream(FormatContextPtr& formatContext, int32_t streamID, CodecContextPtr& codecContext,
        uint32_t bufferLength) noexcept;

    ~Stream() = default;

    Stream(const Stream& other) = delete;

    Stream(Stream&& other) noexcept = delete;

    Stream& operator=(const Stream& other) = delete;

    Stream& operator=(Stream&& other) noexcept = delete;

    /**
     * Gets the width of the video stream.
     * @returns The width.
     */
    [[nodiscard]] uint32_t getWidth() const noexcept;

    /**
     * Gets the height of the video stream.
     * @returns The height.
     */
    [[nodiscard]] uint32_t getHeight() const noexcept;

    /**
     * Gets the display aspect ratio of the video stream.
     * @note This may differ from width/height if stream uses anamorphic pixels.
     * @returns The aspect ratio.
     */
    [[nodiscard]] double getAspectRatio() const noexcept;

    /**
     * Gets total frames in the video stream.
     * @returns The total frames.
     */
    [[nodiscard]] int64_t getTotalFrames() const noexcept;

    /**
     * Gets the duration of the video stream in micro-seconds.
     * @returns The duration.
     */
    [[nodiscard]] int64_t getDuration() const noexcept;

    /**
     * Gets the frame rate (fps) of the video stream.
     * @note This will not be fully accurate when dealing with VFR video streams.
     * @returns The frame rate in frames per second.
     */
    [[nodiscard]] double getFrameRate() const noexcept;

    /**
     * Gets the storage size of each decoded frame in the video stream.
     * @returns The frame size in bytes.
     */
    [[nodiscard]] uint32_t getFrameSize() const noexcept;

    /**
     * Get the next frame in the stream without removing it from stream buffer.
     * @returns The next frame in current stream, or false if an error occured.
     */
    [[nodiscard]] std::variant<bool, std::shared_ptr<Frame>> peekNextFrame() noexcept;

    /**
     * Gets the next frame in the stream and removes it from the buffer.
     * @returns The next frame in current stream, or false if an error occured.
     */
    [[nodiscard]] std::variant<bool, std::shared_ptr<Frame>> getNextFrame() noexcept;

    /**
     * Gets a sequence of frames offset from the current stream position.
     * @param frameSequence The frame sequence. This is a monotonically increasing list of offset indices used to
     * specify which frames to retrieve. e.g. A sequence value of {0, 3, 6} will get the current next frame  as well as
     * the 3rd frame after this and then the third frame after that.
     * @returns A list of frames corresponding to the input sequence, or false if an error occured.
     */
    [[nodiscard]] std::variant<bool, std::vector<std::shared_ptr<Frame>>> getNextFrameSequence(
        const std::vector<int64_t>& frameSequence) noexcept;

    /**
     * Seeks the stream to the given time stamp. If timestamp does not exactly match a frame then the timestamp rounded
     * down to nearest frame is used instead.
     * @param timeStamp The time stamp to seek to (in micro-seconds).
     * @returns True if it succeeds, false if it fails.
     */
    [[nodiscard]] bool seek(int64_t timeStamp) noexcept;

    /**
     * Seeks the stream to the given frame number.
     * @param frame The zero-indexed frame number to seek to.
     * @returns True if it succeeds, false if it fails.
     */
    [[nodiscard]] bool seekFrame(int64_t frame) noexcept;

private:
    std::recursive_mutex m_mutex;

    uint32_t m_bufferLength = 0;                      /**< Length of the ping and pong buffers */
    std::vector<std::shared_ptr<Frame>> m_bufferPing; /**< The primary buffer used to store decoded frames */
    uint32_t m_bufferPingHead =
        0; /**< The position in the ping buffer of the next available frame in the decoded stream. */
    std::vector<std::shared_ptr<Frame>> m_bufferPong; /**< The secondary buffer used to store decoded frames */

    FormatContextPtr m_formatContext;
    int32_t m_index = -1; /**< Zero-based index of the video stream  */
    CodecContextPtr m_codecContext;

    int64_t m_startTimeStamp = 0;     /**< PTS of the first frame in the stream time base */
    bool m_frameSeekSupported = true; /**< True if frame seek supported */
    int64_t m_totalFrames = 0;        /**< Stream video duration in frames */
    int64_t m_totalDuration = 0;      /**< Stream video duration in microseconds (AV_TIME_BASE) */

    /**
     * Convert a time value represented in microseconds (AV_TIME_BASE) to the stream timebase.
     * @param time The time in microseconds (AV_TIME_BASE).
     * @return The converted time stamp.
     */
    [[nodiscard]] int64_t timeToTimeStamp(int64_t time) const noexcept;

    /**
     * Convert a stream timebase to a time value represented in microseconds (AV_TIME_BASE).
     * @param timeStamp The time stamp represented in the streams internal time base.
     * @return The converted time.
     */
    [[nodiscard]] int64_t timeStampToTime(int64_t timeStamp) const noexcept;

    /**
     * Convert a zero-based frame number to the stream timebase.
     * @note This will not be fully accurate when dealing with VFR video streams.
     * @param frame The zero-based frame number
     * @return The converted time stamp.
     */
    [[nodiscard]] int64_t frameToTimeStamp(int64_t frame) const noexcept;

    /**
     * Convert stream based time stamp to an equivalent zero-based frame number.
     * @note This will not be fully accurate when dealing with VFR video streams.
     * @param timeStamp The time stamp represented in the streams internal time base.
     * @return The converted frame index.
     */
    [[nodiscard]] int64_t timeStampToFrame(int64_t timeStamp) const noexcept;

    /**
     * Convert a zero-based frame number to time value represented in microseconds (AV_TIME_BASE).
     * @note This will not be fully accurate when dealing with VFR video streams.
     * @param frame The zero-based frame number
     * @return The converted time.
     */
    [[nodiscard]] int64_t frameToTime(int64_t frame) const noexcept;

    /**
     * Convert a time value represented in microseconds (AV_TIME_BASE) to a zero-based frame number.
     * @note This will not be fully accurate when dealing with VFR video streams.
     * @param time The time in microseconds (AV_TIME_BASE).
     * @return The converted frame index.
     */
    [[nodiscard]] int64_t timeToFrame(int64_t time) const noexcept;

    /**
     * Decodes the next block of frames into the pong buffer. Once complete swaps the ping/pong buffers.
     * @returns True if it succeeds, false if it fails.
     */
    [[nodiscard]] bool decodeNextBlock() noexcept;

    /**
     * Decodes any frames currently pending in the decoder.
     * @returns True if it succeeds, false if it fails.
     */
    [[nodiscard]] bool decodeNextFrames() noexcept;

    /**
     * Pops the next available frame from the buffer.
     * @note This requires that peekNextFrame() be called first to ensure there is a valid frame to pop.
     */
    void popFrame() noexcept;

    /**
     * Seeks the stream to the given time stamp.
     * @param timeStamp The time stamp to seek to (in micro-seconds).
     * @param recursed  True if function has recursed into itself.
     * @returns True if it succeeds, false if it fails.
     */
    [[nodiscard]] bool seekInternal(int64_t timeStamp, bool recursed) noexcept;

    /**
     * Seeks the stream to the given frame number.
     * @param frame The zero-indexed frame number to seek to.
     * @param recursed  True if function has recursed into itself.
     * @returns True if it succeeds, false if it fails.
     */
    [[nodiscard]] bool seekFrameInternal(int64_t frame, bool recursed) noexcept;

    /**
     * Return the maximum number of input frames needed by this stream's codec before it can produce output.
     * @note We expect to have to wait this many frames to receive output; any more and a decode stall is detected.
     * @returns The codec delay.
     */
    [[nodiscard]] int32_t getCodecDelay() const noexcept;

    /**
     * Gets stream start time in the stream timebase.
     * @returns The stream start time.
     */
    [[nodiscard]] int64_t getStreamStartTime() const noexcept;

    /**
     * Gets total number of frames in a stream.
     * @returns The stream frames.
     */
    [[nodiscard]] int64_t getStreamFrames() const noexcept;

    /**
     * Gets the duration of a stream represented in microseconds (AV_TIME_BASE).
     * @returns The duration.
     */
    [[nodiscard]] int64_t getStreamDuration() const noexcept;
};
} // namespace FfFrameReader