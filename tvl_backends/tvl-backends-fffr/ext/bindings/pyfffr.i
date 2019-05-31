%module(directors="1") pyfffr

%{
#include "TvFFFrameReader.h"
#include "MemManager.h"
%}

%feature("director") MemManager;

%typemap(directorout) uint8_t* MemManager::allocate %{
    $result = (uint8_t*)PyInt_AsLong($1);
%}

%typemap(out) int64_t %{
    $result = PyInt_FromLong($1);
%}

%include <std_string.i>
%include "TvFFFrameReader.h"
%include "MemManager.h"
