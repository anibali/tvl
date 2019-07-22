# Torch Video Loader (TVL)

**DISCLAIMER: This project is currently a work in progress.**

TVL is a video loading library with a common interface for decoding on the GPU and CPU. Video
frames are returned as [PyTorch](https://pytorch.org/) tensors, ready for use with a computer
vision model.

```python
import torch
import tvl

# Create a VideoLoader for the video file 'my_video.mkv' that will decode frames as float tensors
# using the first CUDA-enabled GPU device ('cuda:0').
vl = tvl.VideoLoader('my_video.mkv', 'cuda:0', dtype=torch.float32)
# Request three frames by index. Note that the return value is an iterator, and the frames may
# be lazy-loaded.
frames_iter = vl.select_frames([24, 26, 25])
# Force all of the frames to be decoded by creating a list from the iterator. The result will
# be a list of torch.cuda.FloatTensor objects, with shape [3 x H x W].
frames = list(frames_iter)
```


## Requirements

* ffmpeg 4
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

My recommendation is to run the video loader in a single background thread (**note:** using more
than one thread can cause a deadlock, depending on the backend). This enables background loading of
video frames in parallel with your programming doing some other work (eg. training a model).
See [`examples/async_dataloading.py`](examples/async_dataloading.py).


### Backends

| Backend class               | Supported devices |
|-----------------------------|-------------------|
| FffrBackend (recommended)   | cpu,cuda          |
| NvdecBackend                | cuda              |
| PyAvBackend                 | cpu               |
| OpenCvBackend               | cpu               |

If you wanted to install `tvl` with support for the FFFR backend you would install the
package like so:

```bash
$ pip install "tvl[FffrBackend]"
```


### Optimisation tweaks

When you call `select_frames` to read a bunch of frames, TVL must decide when to read frames
sequentially and when to seek. Sometimes reading frames sequentially and discarding unneeded frames
is faster than seeking, and sometimes it isn't. The minimum distance between frames for which
seeking is faster than reading sequentially depends on a number of file-specific factors, such as
GoP size. Unfortunately, these factors can't be inferred automatically on the fly.

In TVL you can manually configure the threshold value for triggering seeks using the
`seek_threshold` backend option.

```python
import tvl
# Provide a hint that if frames are more than 3 frames apart, a seek should be triggered.
vl = tvl.VideoLoader('my_video.mkv', 'cpu', backend_opts={'seek_threshold': 3})
```

If you expect to be reading a lot of videos that are encoded in a similar way, we recommend
benchmarking a range of `seek_threshold` values to find which is fastest.


### Limitations

* GPU support is only available for NVIDIA cards
* Decoding only, no encoding/transcoding
