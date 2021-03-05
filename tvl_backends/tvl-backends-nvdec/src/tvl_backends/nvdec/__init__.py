from functools import lru_cache

import cupy

import tvlnv
from tvl.backend import Backend, BackendFactory


class CupyMemManager(tvlnv.MemManager):
    """MemManager implementation which allocates cupy arrays."""

    def __init__(self):
        super().__init__()
        self.tensors = {}

    def clear(self):
        self.tensors.clear()

    def get_mem_type(self):
        return tvlnv.MEM_TYPE_CUDA

    def allocate(self, size):
        tensor = cupy.empty(size, dtype=cupy.uint8)
        ptr = tensor.__cuda_array_interface__["data"][0]
        self.tensors[ptr] = tensor
        return ptr


def nv12_to_rgb(planar_yuv, h, w):
    """Converts planar YUV pixel data in NV12 format to RGB.

    Args:
        planar_yuv (cupy.array): Planar YUV pixels in [0, 255] value range.
        h: Height of the image.
        w: Width of the image.

    Returns:
        cupy.array: RGB pixels in [0, 255] value range.
    """
    print(planar_yuv.shape)
    y = planar_yuv[: w * h].reshape(h, w, -1)
    u = planar_yuv[w * h : 2 * (w * h)].reshape(h, w, -1)
    v = planar_yuv[2 * (w * h) :].reshape(h, w, -1)
    y -= 16
    u -= 128
    v -= 128
    rgb = cupy.concatenate(
        (
            1.164 * y + 1.596 * v,
            1.164 * y - 0.392 * u - 0.813 * v,
            1.164 * y + 2.017 * u,
        ),
        -1,
    ).astype(cupy.uint8)

    return rgb


class NvdecBackend(Backend):
    def __init__(
        self, filename, device, dtype, *, seek_threshold=3, out_width=0, out_height=0
    ):
        super().__init__(filename, device, dtype, seek_threshold, out_width, out_height)
        mem_manager = CupyMemManager()
        # Disown mem_manager, since TvlnvFrameReader will be responsible for deleting it.
        mem_manager = mem_manager.__disown__()

        self.mem_manager = mem_manager
        self.frame_reader = tvlnv.TvlnvFrameReader(
            mem_manager, self.filename, 0, out_width, out_height
        )

    @property
    def duration(self):
        return self.frame_reader.get_duration()

    @property
    def frame_rate(self):
        return self.frame_reader.get_frame_rate()

    @property
    def n_frames(self):
        return self.frame_reader.get_number_of_frames()

    @property
    def width(self):
        return self.frame_reader.get_width()

    @property
    def height(self):
        return self.frame_reader.get_height()

    def seek(self, time_secs):
        self.frame_reader.seek(time_secs)

    def read_frame(self):
        result = self.frame_reader.read_frame()
        if result is None:
            raise EOFError()
        data_ptr = int(result)
        planar_yuv = self.mem_manager.tensors[data_ptr]
        width = self.frame_reader.get_width()
        height = self.frame_reader.get_height()
        rgb = nv12_to_rgb(planar_yuv, height, width)
        return self._postprocess_frame(rgb)


class NvdecBackendFactory(BackendFactory):
    def create(self, filename, device, dtype, backend_opts=None) -> NvdecBackend:
        if backend_opts is None:
            backend_opts = {}
        return NvdecBackend(filename, device, dtype, **backend_opts)
