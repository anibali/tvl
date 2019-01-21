#include "TvlnvFrameReader.h"

#include "stdio.h"
#include "Utils/NvCodecUtils.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


TvlnvFrameReader::TvlnvFrameReader(MemManager* mem_manager, std::string filename, int gpu_index, Rect* crop_rect, Dim* resize_dim)
    : _mem_manager(mem_manager), _filename(filename)
{
    CheckInputFile(filename.c_str());

    ck(cuInit(0));
    CUdevice cu_device = 0;
    ck(cuDeviceGet(&cu_device, gpu_index));

    // char szDeviceName[80];
    // ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cu_device));
    // printf("GPU in use: %s\n", szDeviceName);

    ck(cuCtxCreate(&_cu_context, CU_CTX_SCHED_BLOCKING_SYNC, cu_device));
    _mem_manager->cu_context = _cu_context;

    _demuxer = new FFmpegDemuxer(_filename.c_str());
    _decoder = new NvDecoder(_cu_context, _demuxer->GetWidth(), _demuxer->GetHeight(),
                             _mem_manager, FFmpeg2NvCodecId(_demuxer->GetVideoCodec()),
                             NULL, false, crop_rect, resize_dim);
}

TvlnvFrameReader::~TvlnvFrameReader() {
    delete _decoder;
    delete _demuxer;
    _mem_manager->cu_context = NULL;
    delete _mem_manager;
    cuCtxDestroy(_cu_context);
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

void TvlnvFrameReader::seek(float time_secs) {
    _demuxer->Seek(time_secs);
}

uint8_t* TvlnvFrameReader::read_frame() {
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t *pVideo = NULL;
    uint8_t **ppFrame = NULL;

    do {
        _demuxer->Demux(&pVideo, &nVideoBytes);
        _decoder->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

//        if (!nFrame && nFrameReturned)
//            LOG(INFO) << _decoder->GetVideoInfo();

        nFrame += nFrameReturned;
    } while(nVideoBytes && nFrameReturned < 1);

    if(nFrameReturned < 1) {
        return NULL;
    }

    return ppFrame[0];
}
