from functools import lru_cache

import torch

import tvlnv
from tvl.backend import Backend, BackendFactory


class TorchMemManager(tvlnv.MemManager):
    """MemManager implementation which allocates Torch tensors."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tensors = {}

    def clear(self):
        self.tensors.clear()

    def get_mem_type(self):
        if self.device.type == 'cuda':
            return tvlnv.MEM_TYPE_CUDA
        return tvlnv.MEM_TYPE_HOST

    def allocate(self, size):
        tensor = torch.empty(size, dtype=torch.uint8, device=self.device)
        ptr = tensor.data_ptr()
        self.tensors[ptr] = tensor
        return ptr


@lru_cache(8)
def _nv12_conv_consts(device):
    const1 = torch.tensor([6.258931e-3, -1.536320e-3, 7.910723e-3], device=device).view(3, 1, 1)
    const2 = torch.tensor([-0.8742, 0.5316706, -1.0856313], device=device).view(3, 1, 1)
    return const1, const2


def nv12_to_rgb(planar_yuv, h, w):
    """Converts planar YUV pixel data in NV12 format to RGB.

    Args:
        planar_yuv (torch.ByteTensor): Planar YUV pixels in [0, 255] value range.
        h: Height of the image.
        w: Width of the image.

    Returns:
        torch.FloatTensor: RGB pixels in [0, 1] value range.
    """
    rgb = torch.empty([3, h, w], dtype=torch.float32, device=planar_yuv.device)
    # Memory reuse trick
    v, _, u = rgb
    # Extract luma channel
    y = planar_yuv[:w*h].view(h, w).float()
    # Extract and upsample chroma channels
    u.copy_(planar_yuv[w*h::2].view(h//2, 1, w//2, 1).expand(h//2, 2, w//2, 2).contiguous().view(h, w))
    v.copy_(planar_yuv[w*h+1::2].view(h//2, 1, w//2, 1).expand(h//2, 2, w//2, 2).contiguous().view(h, w))
    # YUV [0, 255] to RGB [0, 1]
    torch.add(u, v, alpha=2.075161, out=rgb[1])
    const1, const2 = _nv12_conv_consts(str(rgb.device))
    rgb.mul_(const1)
    torch.add(rgb, y, alpha=4.566207e-3, out=rgb)
    rgb.add_(const2)

    return rgb.clamp_(0, 1)


class NvdecBackend(Backend):
    def __init__(self, filename, device, dtype, *, seek_threshold=3, out_width=0, out_height=0):
        super().__init__(filename, device, dtype, seek_threshold, out_width, out_height)
        assert self.device.type == 'cuda'
        mem_manager = TorchMemManager(self.device)
        # Disown mem_manager, since TvlnvFrameReader will be responsible for deleting it.
        mem_manager = mem_manager.__disown__()

        self.mem_manager = mem_manager
        self.frame_reader = tvlnv.TvlnvFrameReader(mem_manager, self.filename, self.device.index,
                                                   out_width, out_height)

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
