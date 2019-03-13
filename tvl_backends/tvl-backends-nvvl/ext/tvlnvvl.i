%module(directors="1") tvlnvvl

%{
#include "PictureSequence.h"
#include "VideoLoader.h"
#include "VideoInfo.h"
%}

%typemap(in) uint16_t {
    $1 = (uint16_t)PyInt_AsLong($input);
}

%typemap(out) uint16_t {
    $result = PyInt_FromLong($1);
}

%typemap(in) uint8_t {
    $1 = (uint8_t)PyInt_AsLong($input);
}

%typemap(out) uint8_t {
    $result = PyInt_FromLong($1);
}

%typemap(in) int64_t {
    $1 = (int64_t)PyInt_AsLong($input);
}

%typemap(out) int64_t {
    $result = PyInt_FromLong($1);
}

%typemap(in) void* {
    $1 = (void*)PyInt_AsLong($input);
}

%typemap(out) void* {
    $result = PyInt_FromLong((size_t)$1);
}

%include <std_string.i>
%include "nvvl/PictureSequence.h"
%include "nvvl/VideoLoader.h"
%include "VideoInfo.h"
