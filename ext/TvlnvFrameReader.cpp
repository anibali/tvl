#include "TvlnvFrameReader.h"

#include "stdio.h"
#include "Utils/NvCodecUtils.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


TvlnvFrameReader::TvlnvFrameReader(MemManager* mem_manager, std::string filename, int gpu_index)
    : _mem_manager(mem_manager), _filename(filename)
{
    CheckInputFile(filename.c_str());

    ck(cuInit(0));
    ck(cuDeviceGet(&_cu_device, gpu_index));

    // char szDeviceName[80];
    // ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), _cu_device));
    // printf("GPU in use: %s\n", szDeviceName);

    ck(cuDevicePrimaryCtxRetain(&_cu_context, _cu_device));
    _mem_manager->cu_context = _cu_context;

    _demuxer = new FFmpegDemuxer(_filename.c_str());
    _decoder = new NvDecoder(_cu_context, _demuxer->GetWidth(), _demuxer->GetHeight(),
                             _mem_manager, FFmpeg2NvCodecId(_demuxer->GetVideoCodec()));
}

TvlnvFrameReader::~TvlnvFrameReader() {
    delete _decoder;
    delete _demuxer;
    _mem_manager->cu_context = NULL;
    delete _mem_manager;
    cuDevicePrimaryCtxRelease(_cu_device);
}

std::string TvlnvFrameReader::get_filename() {
    return _filename;
}

int TvlnvFrameReader::get_width() {
    return _decoder->GetWidth();
}

int TvlnvFrameReader::get_height() {
    return _decoder->GetHeight();
}

int TvlnvFrameReader::get_frame_size() {
    return _decoder->GetFrameSize();
}

double TvlnvFrameReader::get_duration() {
    return _demuxer->GetDuration();
}

double TvlnvFrameReader::get_frame_rate() {
    return _demuxer->GetFrameRate();
}

int64_t TvlnvFrameReader::get_number_of_frames() {
    return _demuxer->GetNumberOfFrames();
}

void TvlnvFrameReader::seek(float time_secs) {
    _seek_pts = _demuxer->SecsToPts(time_secs);
    _demuxer->Seek(_seek_pts);
    // Clear all buffered frames.
    while(!frame_buf.empty()) {
        frame_buf.pop();
    }
    // Reset the decoder.
    delete _decoder;
    _decoder = new NvDecoder(_cu_context, _demuxer->GetWidth(), _demuxer->GetHeight(),
                             _mem_manager, FFmpeg2NvCodecId(_demuxer->GetVideoCodec()));
}

uint8_t* TvlnvFrameReader::read_frame() {
    if(!frame_buf.empty()) {
        uint8_t* pFrame = frame_buf.front();
        frame_buf.pop();
        return pFrame;
    }

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t *pVideo = NULL;
    uint8_t **ppFrame = NULL;
    int64_t *pTimestamp;

    bool seeking = _seek_pts != AV_NOPTS_VALUE;
    do {
        _demuxer->Demux(&pVideo, &nVideoBytes);
        _decoder->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned, CUVID_PKT_ENDOFPICTURE, &pTimestamp, _demuxer->pkt_pts);

//        if (!nFrame && nFrameReturned)
//            LOG(INFO) << _decoder->GetVideoInfo();
        nFrame += nFrameReturned;

        for (int i = 0; i < nFrameReturned; i++)
        {
            if(pTimestamp[i] >= _seek_pts) {
                seeking = false;
            }
        }
    } while(nVideoBytes && (seeking || nFrameReturned < 1));

    if(nFrameReturned < 1) {
        return NULL;
    }

    for(int i = 0; i < nFrameReturned; ++i) {
        if(pTimestamp[i] >= _seek_pts) {
            frame_buf.push(ppFrame[i]);
        }
    }

    _seek_pts = AV_NOPTS_VALUE;

    if(!frame_buf.empty()) {
        uint8_t* pFrame = frame_buf.front();
        frame_buf.pop();
        return pFrame;
    }

    return NULL;
}
