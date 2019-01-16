%module(directors="1") tvlnv

%{
#include "TvlnvFrameReader.h"
#include "MemManager.h"
%}

%feature("director") MemManager;

%typemap(directorout) uint8_t* MemManager::allocate %{
  $result = (uint8_t*)PyInt_AsLong($1);
%}

%include <std_string.i>
%include "TvlnvFrameReader.h"
%include "MemManager.h"
