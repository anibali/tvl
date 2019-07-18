import torch

from tvl_backends.nvdec import nv12_to_rgb


def test_nv12_to_rgb():
    w = 3840
    h = 2160
    nv12 = torch.empty(int(w * h * 1.5), device='cuda:0', dtype=torch.uint8)
    for i in range(100):
        nv12.random_(0, 256)
        rgb = nv12_to_rgb(nv12, h, w)
        assert rgb.shape == (3, h, w)
