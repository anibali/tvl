import tvlnv

import torch
import numpy as np
import ctypes as C
import PIL.Image
from time import time


class TorchMemManager(tvlnv.MemManager):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tensors = []

    def clear(self):
        self.tensors.clear()

    def get_mem_type(self):
        if self.device.type == 'cuda':
            return tvlnv.MEM_TYPE_CUDA
        return tvlnv.MEM_TYPE_HOST

    def allocate(self, size):
        tensor = torch.empty(size, dtype=torch.uint8, device=self.device)
        self.tensors.append(tensor)
        return tensor.data_ptr()


# def test_torch_mem_manager():
#     filename = '/data/diving/processed/brisbane2016/DAY 4_6th May 2016_Friday/Day 4_Block 2.SCpkg/video.mkv'
#     mm = TorchMemManager(torch.device('cuda:0'))
#     mm.__disown__()
#     fr = tvlnv.TvlnvFrameReader(mm, filename)
#
#     fr.test_callback()


def test_tvl():
    print()
    # filename = '/home/aiden/Projects/PyTorch/tvl/tests/data/lines.mkv'
    # filename = '/home/aiden/Videos/bengio_twitter_boston_20160512/bengio_twitter_boston_20160512.mp4'
    filename = '/data/diving/processed/brisbane2016/DAY 4_6th May 2016_Friday/Day 4_Block 2.SCpkg/video.mkv'
    mm = TorchMemManager(torch.device('cuda:0'))
    mm.__disown__()
    fr = tvlnv.TvlnvFrameReader(mm, filename)
    frame_ptr = int(fr.read_frame())

    w = 1920
    h = 1080
    bytes_per_pixel = 1.5

    frame_bytes = int(w * h * bytes_per_pixel)

    start = time()
    # NOTE: This copies data due to some internal quirk.
    na = np.ctypeslib.as_array(C.cast(frame_ptr, C.POINTER(C.c_uint8)), (frame_bytes,))
    b = torch.from_numpy(na).cuda()

    # Format is NV12 (I think)
    yuv = b.cpu()
    y = yuv[:w*h].view(h, w).numpy()
    uv = yuv[w*h:]
    u = uv[::2].view(h//2, w//2).numpy().repeat(2, axis=0).repeat(2, axis=1)
    v = uv[1::2].view(h//2, w//2).numpy().repeat(2, axis=0).repeat(2, axis=1)
    img = PIL.Image.fromarray(np.stack([y, u, v], axis=-1), 'YCbCr')
    # img.show()

    # ~2.4 seconds
    print(time() - start)
