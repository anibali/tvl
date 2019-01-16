import _tvlnv

import torch
import numpy as np
import ctypes as C
import PIL.Image
from time import time


def test_tvl():
    torch.tensor([1,2]).cuda()

    assert 2 + 2 == 4
    print()

    # fr = _tvlnv.new_TvlnvFrameReader('/home/aiden/Projects/PyTorch/tvl/tests/data/lines.mkv')
    # fr = _tvlnv.new_TvlnvFrameReader('/home/aiden/Videos/bengio_twitter_boston_20160512/bengio_twitter_boston_20160512.mp4')
    fr = _tvlnv.new_TvlnvFrameReader('/data/diving/processed/brisbane2016/DAY 4_6th May 2016_Friday/Day 4_Block 2.SCpkg/video.mkv')
    frame_ptr = int(_tvlnv.TvlnvFrameReader_read_frame(fr))

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

    _tvlnv.delete_TvlnvFrameReader(fr)

    print(time() - start)
