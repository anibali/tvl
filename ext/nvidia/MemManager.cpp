#include "MemManager.h"

#include "NvDecoder/NvDecoder.h"


uint8_t* HostMemManager::allocate(size_t size) {
    uint8_t* pFrame = new uint8_t[size];
    _allocated.push_back(pFrame);
    return pFrame;
}

void HostMemManager::clear() {
    uint8_t* pFrame;
    while(!_allocated.empty())
    {
        pFrame = _allocated.back();
        _allocated.pop_back();
        delete[] pFrame;
    }
}

MemType HostMemManager::get_mem_type() {
    return MEM_TYPE_HOST;
}

uint8_t* CuMemManager::allocate(size_t size) {
    uint8_t* pFrame = NULL;
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(cu_context));
    CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr *)&pFrame, size));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    _allocated.push_back(pFrame);
    return pFrame;
}

void CuMemManager::clear() {
    uint8_t* pFrame;
    while(!_allocated.empty())
    {
        pFrame = _allocated.back();
        _allocated.pop_back();
        CUDA_DRVAPI_CALL(cuCtxPushCurrent(cu_context));
        CUDA_DRVAPI_CALL(cuMemFree((CUdeviceptr)pFrame));
        CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    }
}

MemType CuMemManager::get_mem_type() {
    return MEM_TYPE_CUDA;
}
