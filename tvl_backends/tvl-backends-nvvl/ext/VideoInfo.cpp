#include "VideoInfo.h"

#include <sstream>
extern "C" {
#include <libavformat/avformat.h>
}


VideoInfo::VideoInfo(std::string filename) {
    av_register_all();

    AVFormatContext* fmt_ctx = nullptr;
    auto ret = avformat_open_input(&fmt_ctx, filename.c_str(), NULL, NULL);
    if (ret < 0) {
        std::stringstream err;
        err << "Could not open file " << filename;
        throw std::runtime_error(err.str());
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        throw std::runtime_error(std::string("Could not find stream information in ")
                                 + filename);
    }

    auto vid_stream_idx_ = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO,
                                               -1, -1, nullptr, 0);
    if (vid_stream_idx_ < 0) {
        throw std::runtime_error(std::string("Could not find video stream in ") + filename);
    }

    auto stream = fmt_ctx->streams[vid_stream_idx_];

    _width = stream->codecpar->width;
    _height = stream->codecpar->height;
    _duration = fmt_ctx->duration / (double)AV_TIME_BASE;
    _frame_rate = av_q2d(av_guess_frame_rate(fmt_ctx, stream, NULL));
    _n_frames = stream->nb_frames;
    if(_n_frames <= 0) {
        _n_frames = (int64_t)(_duration * _frame_rate);
    }

    avformat_close_input(&fmt_ctx);
}

std::string VideoInfo::get_filename() {
    return _filename;
}

int VideoInfo::get_width() {
    return _width;
}

int VideoInfo::get_height() {
    return _height;
}

double VideoInfo::get_duration() {
    return _duration;
}

double VideoInfo::get_frame_rate() {
    return _frame_rate;
}

int64_t VideoInfo::get_number_of_frames() {
    return _n_frames;
}
