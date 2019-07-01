%module(directors="1") pyfffr

%{
#include "TvFFFrameReader.h"
#include "ImageAllocator.h"
%}

%feature("director") ImageAllocator;

%typemap(directorout) void* ImageAllocator::allocate_frame %{
    $result = (void*)PyInt_AsLong($1);
%}

%typemap(out) int64_t %{
    $result = PyInt_FromLong($1);
%}

%typemap(in) int64_t* %{
    $1 = (int64_t*)PyInt_AsLong($input);
%}

%typemap(in) uint8_t** %{
    $1 = (uint8_t**)PyInt_AsLong($input);
%}

/*
  Catch all C++ exceptions and rethrow as Python exceptions. There's a good explanation of how
  this mechanism works here:
  https://github.com/trilinos/Trilinos/blob/master/packages/PyTrilinos/doc/DevelopersGuide/ExceptionHandling.txt
*/
%include <exception.i>
%exception {
    try {
        $action
    }
    SWIG_CATCH_STDEXCEPT
    catch(...) {
        SWIG_exception(SWIG_UnknownError, "Unknown exception");
    }
}

%include <std_string.i>
%include "TvFFFrameReader.h"
%include "ImageAllocator.h"
