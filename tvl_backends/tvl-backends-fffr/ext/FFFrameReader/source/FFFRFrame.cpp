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
#include "FFFRFrame.h"

#include <algorithm>

extern "C" {
#include <libavutil/frame.h>
}

using namespace std;

namespace FfFrameReader {
Frame::FramePtr::FramePtr(AVFrame* frame) noexcept
    : m_frame(frame)
{}

Frame::FramePtr::~FramePtr() noexcept
{
    if (m_frame != nullptr) {
        av_frame_free(&m_frame);
    }
}

Frame::FramePtr::FramePtr(FramePtr&& other) noexcept
    : m_frame(other.m_frame)
{
    other.m_frame = nullptr;
}

Frame::FramePtr& Frame::FramePtr::operator=(FramePtr& other) noexcept
{
    m_frame = other.m_frame;
    other.m_frame = nullptr;
    return *this;
}

Frame::FramePtr& Frame::FramePtr::operator=(FramePtr&& other) noexcept
{
    m_frame = other.m_frame;
    other.m_frame = nullptr;
    return *this;
}

AVFrame*& Frame::FramePtr::operator*() noexcept
{
    return m_frame;
}

const AVFrame* Frame::FramePtr::operator*() const noexcept
{
    return m_frame;
}

AVFrame*& Frame::FramePtr::operator->() noexcept
{
    return m_frame;
}

const AVFrame* Frame::FramePtr::operator->() const noexcept
{
    return m_frame;
}

Frame::Frame(FramePtr& frame, const int64_t timeStamp, const int64_t frameNum) noexcept
    : m_frame(move(frame))
    , m_timeStamp(timeStamp)
    , m_frameNum(frameNum)
{}

int64_t Frame::getTimeStamp() const noexcept
{
    return m_timeStamp;
}

int64_t Frame::getFrameNumber() const noexcept
{
    return m_frameNum;
}

uint8_t* const* Frame::getFrameData() const noexcept
{
    return m_frame->data;
}

uint32_t Frame::getWidth() const noexcept
{
    return m_frame->width;
}

uint32_t Frame::getHeight() const noexcept
{
    return m_frame->height;
}

double Frame::getAspectRatio() const noexcept
{
    // TODO: Handle this with anamorphic
    return static_cast<double>(getWidth()) / static_cast<double>(getHeight());
}

int Frame::getPixelFormat() const noexcept
{
    return m_frame->format;
}
} // namespace FfFrameReader