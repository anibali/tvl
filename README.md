# Torch Video Loader (TVL)

TVL is a video loading library with a common interface for decoding on the GPU and CPU. Video
frames are returned as [PyTorch](https://pytorch.org/) tensors, ready for use with a computer
vision model.

Current limitations:

* H.264 video codec only
* NVIDIA GPUs only

## Requirements

* ffmpeg 3.4
* SWIG 3
* NVIDIA drivers >= 396.24

## Building

```bash
rm -rf build && python setup.py build_ext --inplace && pytest -s
```

Dockerised version:

```bash
docker build -t tvl . && docker run --rm -it tvl
```

## TODO

* Some of the stuff in the TvlnvFrameReader constructor should probably be moved to
  a global init function of some variety.
