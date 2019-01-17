import tvlnv

import torch
import numpy as np
import PIL.Image
from time import time
import ctypes as C


# VIDEO_FILENAME = '/home/aiden/Projects/PyTorch/tvl/tests/data/lines.mkv'
VIDEO_FILENAME = '/home/aiden/Videos/bengio_twitter_boston_20160512/bengio_twitter_boston_20160512.mp4'
# VIDEO_FILENAME = '/data/diving/processed/brisbane2016/DAY 4_6th May 2016_Friday/Day 4_Block 2.SCpkg/video.mkv'


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


def test_torch_mem_manager():
    mm = TorchMemManager(torch.device('cuda:1'))
    mm.__disown__()

    fr = tvlnv.TvlnvFrameReader(mm, VIDEO_FILENAME, mm.device.index)
    fr.seek(120.0)
    start = time()
    frame_ptr = int(fr.read_frame())

    w = fr.get_width()
    h = fr.get_height()

    b = mm.tensors[frame_ptr]

    # Format is NV12 (I think)
    yuv = b.cpu()
    y = yuv[:w*h].view(h, w).numpy()
    uv = yuv[w*h:]
    u = uv[::2].view(h//2, w//2).numpy().repeat(2, axis=0).repeat(2, axis=1)
    v = uv[1::2].view(h//2, w//2).numpy().repeat(2, axis=0).repeat(2, axis=1)
    img = PIL.Image.fromarray(np.stack([y, u, v], axis=-1), 'YCbCr')
    img.show()

    print(time() - start)



# def test_tvl():
#     # mm = TorchMemManager(torch.device('cuda:0'))
#     # mm.__disown__()
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
