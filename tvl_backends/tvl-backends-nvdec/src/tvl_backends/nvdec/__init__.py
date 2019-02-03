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
    y.mul_(4.566207e-3)
    torch.add(u, 2.075161, v, out=rgb[1]).mul_(-1.536320e-3).add_(y).add_(0.5316706)  # Green
    v.mul_(6.258931e-3).add_(y).add_(-0.8742)  # Red
    u.mul_(7.910723e-3).add_(y).add_(-1.0856313)  # Blue
    return rgb.clamp_(0, 1)


class NvdecBackend(Backend):
    def __init__(self, filename, device):
        device = torch.device(device)
        mem_manager = TorchMemManager(device)
        mem_manager.__disown__()
        self.mem_manager = mem_manager

        self.frame_reader = tvlnv.TvlnvFrameReader(mem_manager, filename, device.index)

    @property
    def duration(self):
        return self.frame_reader.get_duration()

    @property
    def frame_rate(self):
        return self.frame_reader.get_frame_rate()

    @property
    def n_frames(self):
        return self.frame_reader.get_number_of_frames()

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
        return rgb


class NvdecBackendFactory(BackendFactory):
    def create(self, filename, device) -> NvdecBackend:
        return NvdecBackend(filename, device)
