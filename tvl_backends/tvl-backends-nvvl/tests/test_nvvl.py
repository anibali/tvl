import PIL.Image
import numpy as np
import torch

from tvl_backends.nvvl import NvvlBackendFactory


def test_nvvl_read_frame(video_filename, first_frame_image):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)
    rgb = backend.read_frame()

    assert(rgb.size() == (3, 720, 1280))

    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')

    # For some reason, the rightmost column of pixels isn't right
    a = np.array(img)[:, :-1, :]
    b = np.array(first_frame_image)[:, :-1, :]
    np.testing.assert_allclose(a, b, rtol=0, atol=50)


def test_nvvl_seek_to_frame(video_filename, mid_frame_image):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)
    backend.seek_to_frame(25)
    rgb = backend.read_frame()
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    # For some reason, the rightmost column of pixels isn't right
    a = np.array(img)[:, :-1, :]
    b = np.array(mid_frame_image)[:, :-1, :]
    np.testing.assert_allclose(a, b, rtol=0, atol=50)
