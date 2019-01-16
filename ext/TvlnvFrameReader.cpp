#include "TvlnvFrameReader.h"

#include "stdio.h"
#include "Utils/NvCodecUtils.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


TvlnvFrameReader::TvlnvFrameReader(std::string filename)
    : _filename(filename)
{
    const int gpu_index = 0;
    const bool keep_frame_on_gpu = true;

    CheckInputFile(filename.c_str());

    ck(cuInit(0));
    CUdevice cu_device = 0;
    ck(cuDeviceGet(&cu_device, gpu_index));

    // char szDeviceName[80];
    // ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cu_device));
    // printf("GPU in use: %s\n", szDeviceName);

    ck(cuCtxCreate(&_cu_context, CU_CTX_SCHED_BLOCKING_SYNC, cu_device));
    _demuxer = new FFmpegDemuxer(_filename.c_str());
    _decoder = new NvDecoder(_cu_context, _demuxer->GetWidth(), _demuxer->GetHeight(),
                             keep_frame_on_gpu, FFmpeg2NvCodecId(_demuxer->GetVideoCodec()));
    printf("Frame size: %d\n", _demuxer->GetFrameSize());
}

TvlnvFrameReader::~TvlnvFrameReader() {
    delete _decoder;
    delete _demuxer;
    cuCtxDestroy(_cu_context);
}

std::string TvlnvFrameReader::get_filename() {
    return this->_filename;
}

uint8_t* TvlnvFrameReader::read_frame() {
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t *pVideo = NULL;
    uint8_t **ppFrame = NULL;

//    _demuxer->Seek(50.0);

    do {
        _demuxer->Demux(&pVideo, &nVideoBytes);
        _decoder->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

        if (!nFrame && nFrameReturned)
            LOG(INFO) << _decoder->GetVideoInfo();

        nFrame += nFrameReturned;
    } while(nVideoBytes && nFrameReturned < 1);

    if(nFrameReturned < 1) {
        return NULL;
    }

    return ppFrame[0];
}

void TvlnvFrameReader::read_frames() {
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t *pVideo = NULL;
    uint8_t **ppFrame = 0;

    _demuxer->Seek(5.2);

    do {
        _demuxer->Demux(&pVideo, &nVideoBytes);
        _decoder->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

        if (!nFrame && nFrameReturned)
            LOG(INFO) << _decoder->GetVideoInfo();

        nFrame += nFrameReturned;
    } while (nVideoBytes);
    std::cout << "Total frame decoded: " << nFrame << std::endl;
}
