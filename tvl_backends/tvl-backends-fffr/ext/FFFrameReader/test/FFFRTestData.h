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
#include <cstdint>
#include <string>
#include <vector>

struct TestParams
{
    std::string m_fileName;
    int64_t m_width;
    int64_t m_height;
    double m_aspectRatio;
    int64_t m_totalFrames;
    int64_t m_duration;
    double m_frameRate;
    int64_t m_frameTime;
};

static std::vector<TestParams> g_testData = {
    {"data/bbb_sunflower_1080p_30fps_normal.mp4", 1920, 1080, 16.0 / 9.0, 19034, 634466666, 30.0, 33333},
#if 1
    {"data/CLIP0000818_000_77aabf.mkv", 3840, 2160, 16.0 / 9.0, 1116, 44640000, 25.0, 40000},
    {"data/MenMC_50BR_Finals_Canon - Cut.mp4", 3840, 2160, 16.0 / 9.0, 249, 9960000, 25.0, 40000},
    {"data/MenMC_50BR_Finals_Canon.MP4", 3840, 2160, 16.0 / 9.0, 2124, 84960000, 25.0, 40000},
    {"data/MVI_0048.MP4", 3840, 2160, 16.0 / 9.0, 8592, 171840000, 50.0, 20000},
    {"data/Women_400IM_Heat2_Sony.MXF", 3840, 2160, 16.0 / 9.0, 9204, 368160000, 25.0, 40000},
#endif
};