%module(directors="1") tvlnv

%{
#include "TvlnvFrameReader.h"
#include "MemManager.h"
%}

%feature("director") MemManager;

%typemap(directorout) void* MemManager::allocate %{
  $result = (void*)PyInt_AsLong($1);
%}

%include <std_string.i>
%include "TvlnvFrameReader.h"
%include "MemManager.h"
