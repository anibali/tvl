FFFrameReader
=============
[![Github All Releases](https://img.shields.io/github/downloads/Sibras/FFFrameReader/total.svg)](https://github.com/Sibras/FFFrameReader/releases)
[![GitHub release](https://img.shields.io/github/release/Sibras/FFFrameReader.svg)](https://github.com/Sibras/FFFrameReader/releases/latest)
[![GitHub issues](https://img.shields.io/github/issues/Sibras/FFFrameReader.svg)](https://github.com/Sibras/FFFrameReader/issues)
[![license](https://img.shields.io/github/license/Sibras/FFFrameReader.svg)](https://github.com/Sibras/FFFrameReader/blob/master/LICENSE)
[![donate](https://img.shields.io/badge/donate-link-brightgreen.svg)](https://shiftmediaproject.github.io/8-donate/)

## About

This project provides a library the wraps Ffmpeg decoding functionality into a simple to use c++ library.
It is designed to allow for easily retrieving frames from a video file that can then be used as required by a host program.
The library supports software decoding as well as some hardware decoders with basic conversion and processing support.

## Example:
To use this library you first need to create a decoder context that is used to represent the decoding options that
are to be used when decoding files. A default DecoderContext object will use normal software decoding on the CPU and
can be created using the following:
~~~~
DecoderContext context;
~~~~
Decoder contexts can also be optionally passed a set of options (@see DecoderOptions) that can be used to enable
hardware accelerated decoding and various other decoding parameters. For instance to use NVIDIA GPU accelerated
decoding you can create a decoder context as follows:
~~~~
DecoderContext context(DecoderContext::DecodeType::Nvdec);
~~~~
Once a decoder has been created it can then be used to open various files such as:
~~~~
auto ret = context.getStream(fileName);
if (ret.index() == 0) {
    // File opening has failed
}
auto stream = std::get<1>(ret);
~~~~
A stream object can then be used to get information about the opened file (such as resolution, duration etc.) and to
read image frames from the video. To get the next frame in a video you can use the following in a loop:
~~~~
auto ret2 = stream->getNextFrame();
if (ret2.index() == 0) {
    // Failed to get the next frame, there was either an internal error or the end of the file has been reached
}
auto frame = std::get<1>(ret2);
~~~~
This creates a frame object that has the information for the requested frame (such as its time stamp, frame number
etc.). It also has the actual image data itself that can be retrieved as follows:
~~~~
uint8_t* const* data = frame->getFrameData();
~~~~
The format of the data stored in the returned pointer depends on the input video and the decoder options. By default
many videos will be coded using YUV422 which in that case means that data is a pointer to an array of 3 memory
pointers, where each of the 3 pointers points to either the Y,U or V planes respectively.

By default when using hardware decoding the decoded frames will be output in the default memory format of the chosen
decoder. So for instance when using NVDec the output memory will be a CUDA GPU memory pointer.

In addition to just reading frames in sequence from start to finish, a stream object also supports seeking. Seeks can
be performed on either duration time stamps of specific frame numbers as follows:
~~~~
if (!stream->seek(2000)) {
    // Failed to seek to requested time stamp
}
~~~~