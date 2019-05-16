#pragma once

#include <stddef.h>
#include <stdint.h>


class MemManager {
public:
    virtual ~MemManager() {}
    virtual uint8_t* allocate(size_t size) = 0;
    virtual void free(uint8_t* address) = 0;
    virtual void clear() {}
};
