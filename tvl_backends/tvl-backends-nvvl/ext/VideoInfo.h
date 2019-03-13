#pragma once

#include <string>


class VideoInfo
{
public:
    VideoInfo(std::string filename);

    std::string get_filename();
    int get_width();
    int get_height();
    double get_duration();
    double get_frame_rate();
    int64_t get_number_of_frames();

private:
    std::string _filename;
    int _width;
    int _height;
    double _duration;
    double _frame_rate;
    int64_t _n_frames;
};
