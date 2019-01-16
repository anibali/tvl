#pragma once

#include <stddef.h>
#include <cuda.h>
#include <vector>

typedef enum {
    MEM_TYPE_HOST,
    MEM_TYPE_CUDA,
    LENGTH_MEM_TYPE,
} MemType;

class MemManager {
public:
    virtual ~MemManager() {}
    virtual uint8_t* allocate(size_t size) = 0;
    virtual void clear() {}
    virtual MemType get_mem_type() = 0;

    CUcontext cu_context;
};

class HostMemManager: public MemManager {
    virtual uint8_t* allocate(size_t size);
    virtual void clear();
    virtual MemType get_mem_type();

private:
    std::vector<uint8_t*> _allocated;
};

class CuMemManager: public MemManager {
    virtual uint8_t* allocate(size_t size);
    virtual void clear();
    virtual MemType get_mem_type();

private:
    std::vector<uint8_t*> _allocated;
};
