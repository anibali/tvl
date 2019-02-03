%module(directors="1") tvlnv

%{
#include "TvlnvFrameReader.h"
#include "MemManager.h"

static PyObject* py_NVDECException;
%}

%init %{
    py_NVDECException = PyErr_NewException("_tvlnv.NVDECException", NULL, NULL);
    Py_INCREF(py_NVDECException);
    PyModule_AddObject(m, "NVDECException", py_NVDECException);
%}

%pythoncode %{
    NVDECException = _tvlnv.NVDECException
%}

%exception TvlnvFrameReader::TvlnvFrameReader {
    try {
        $function
    } catch(const NVDECException &ex) {
        PyErr_SetString(py_NVDECException, const_cast<char*>(ex.what()));
        return NULL;
    }
}

%feature("director") MemManager;

%typemap(directorout) uint8_t* MemManager::allocate %{
    $result = (uint8_t*)PyInt_AsLong($1);
%}

%typemap(out) int64_t %{
    $result = PyInt_FromLong($1);
%}

%include <std_string.i>
%include "TvlnvFrameReader.h"
%include "nvidia/MemManager.h"

struct Rect {
    int l, t, r, b;
};

struct Dim {
    int w, h;
};
