import tvlnv

import torch
import numpy as np
import PIL.Image
from time import time
import ctypes as C


# VIDEO_FILENAME = '/home/aiden/Projects/PyTorch/tvl/tests/data/lines.mkv'
# VIDEO_FILENAME = '/home/aiden/Videos/bengio_twitter_boston_20160512/bengio_twitter_boston_20160512.mp4'
VIDEO_FILENAME = '/data/diving/processed/brisbane2016/DAY 4_6th May 2016_Friday/Day 4_Block 2.SCpkg/video.mkv'


class TorchMemManager(tvlnv.MemManager):
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
    torch.add(u, 2.075161, v, out=rgb[1]).mul_(-1.536320e-3).add_(y).add_(0.5316706) # Green
    v.mul_(6.258931e-3).add_(y).add_(-0.8742) # Red
    u.mul_(7.910723e-3).add_(y).add_(-1.0856313) # Blue
    return rgb.clamp_(0, 1)


def test_torch_mem_manager():
    mm = TorchMemManager(torch.device('cuda:1'))
    mm.__disown__()

    dim = tvlnv.Dim()
    dim.w = 1280
    dim.h = 720

    fr = tvlnv.TvlnvFrameReader(mm, VIDEO_FILENAME, mm.device.index, resize_dim=dim)

    fr.seek(120.0)
    planar_yuv = mm.tensors[int(fr.read_frame())]

    w = fr.get_width()
    h = fr.get_height()

    start = time()
    rgb = nv12_to_rgb(planar_yuv, h, w)
    print(time() - start)

    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    img.show()



# def test_tvl():
#     mm = tvlnv.CuMemManager()
#     mm.thisown = 0
#     fr = tvlnv.TvlnvFrameReader(mm, VIDEO_FILENAME, gpu_index=1)
#     fr.seek(120.0)
#     start = time()
#     frame_ptr = int(fr.read_frame())
#
#     w = fr.get_width()
#     h = fr.get_height()
#     frame_bytes = fr.get_frame_size()
#
#     # NOTE: This copies data due to some internal quirk.
#     na = np.ctypeslib.as_array(C.cast(frame_ptr, C.POINTER(C.c_uint8)), (frame_bytes,))
#     b = torch.from_numpy(na).cuda()
#
#     # b = mm.tensors[frame_ptr]
#
#     # Format is NV12 (I think)
#     yuv = b.cpu()
#     y = yuv[:w*h].view(h, w).numpy()
#     uv = yuv[w*h:]
#     u = uv[::2].view(h//2, w//2).numpy().repeat(2, axis=0).repeat(2, axis=1)
#     v = uv[1::2].view(h//2, w//2).numpy().repeat(2, axis=0).repeat(2, axis=1)
#     img = PIL.Image.fromarray(np.stack([y, u, v], axis=-1), 'YCbCr')
#     img.show()
#
#     print(time() - start)
