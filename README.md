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


## Parallelism

CUDA and multiprocessing don't mix very well. When multiprocessing is in "fork" mode, CUDA
straight-up [fails to initialise](https://devtalk.nvidia.com/default/topic/973477/-cuda8-0-bug-child-process-forked-after-cuinit-get-cuda_error_not_initialized-on-cuinit-/).
Specifying spawn/forkserver for multiprocessing works, but is ridiculously slow.

My recommendation is to run the video loader in a single background thread. This enables
background loading of video frames in parallel with your programming doing some other work (eg.
training a model). See [`examples/async_dataloading.py`](examples/async_dataloading.py).


## TODO

* Some of the stuff in the TvlnvFrameReader constructor should probably be moved to
  a global init function of some variety.
