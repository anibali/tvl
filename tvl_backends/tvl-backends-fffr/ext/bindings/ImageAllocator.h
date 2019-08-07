#pragma once

#include <stddef.h>
#include <stdint.h>


class ImageAllocator {
public:
    typedef enum {
        UINT8,
        FLOAT32,
    } DataType;

    virtual ~ImageAllocator() {}
    /**
     * Allocate memory for a 3-plane image with 32-byte alignment for each plane.
     */
    virtual void* allocate_frame(int width, int height, int line_size, int alignment) = 0;
    /**
     * Free (or release reference to) previously allocated frame memory.
     */
    virtual void free_frame(void* address) = 0;
    /**
     * Get the frame data type.
     */
    virtual DataType get_data_type() = 0;
    /**
     * Get the GPU device index for memory allocated by this allocator.
     * A return value of -1 means main system memory.
     */
    virtual int get_device_index() = 0;
};
