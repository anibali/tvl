# Torch Video Loader (TVL)

**DISCLAIMER: This project is currently a work in progress.**

TVL is a video loading library with a common interface for decoding on the GPU and CPU. Video
frames are returned as [PyTorch](https://pytorch.org/) tensors, ready for use with a computer
vision model.

```python
import tvl

device = 'cuda:0'  # Use 'cpu' for CPU decoding
vl = tvl.VideoLoader('my_video.mkv', device)
list_of_rgb_tensors = vl.pick_frames([24, 26, 25])
```


## Requirements

* ffmpeg 3.4
* SWIG 3
* NVIDIA drivers >= 396.24


## Building from source

### Build wheels on host

```bash
make dist
```

The wheels for tvl and all backends will be placed in `dist/`.

### Build and run tests with Docker

```bash
docker build -t tvl . && docker run --rm -it tvl
```


## Video player example

A simple Tkinter-based video player is provided in the `examples/` directory. Try it out by running
the following command:

```bash
python examples/video_player.py
```


## Notes


### Parallelism

CUDA and multiprocessing don't mix very well. When multiprocessing is in "fork" mode, CUDA
straight-up [fails to initialise](https://devtalk.nvidia.com/default/topic/973477/-cuda8-0-bug-child-process-forked-after-cuinit-get-cuda_error_not_initialized-on-cuinit-/).
Specifying spawn/forkserver for multiprocessing works, but is ridiculously slow.

My recommendation is to run the video loader in a single background thread. This enables
background loading of video frames in parallel with your programming doing some other work (eg.
training a model). See [`examples/async_dataloading.py`](examples/async_dataloading.py).


### Backends

| Backend class | Supported devices |
|---------------|-------------------|
| NvdecBackend  | cuda              |
| PyAvBackend   | cpu               |
| OpenCvBackend | cpu               |

If you wanted to install `tvl` with support for the NVDEC and PyAV backends you would install the
package like so:

```bash
$ pip install "tvl[NvdecBackend,PyAvBackend]"
```


### Limitations

* GPU support is only available for NVIDIA cards
* Decoding only, no encoding/transcoding
