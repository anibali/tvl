import PIL.Image
import numpy as np
import pytest
import torch

from tvl_backends.nvvl import NvvlBackendFactory


def _assert_same_image(actual_img, expected_img):
    # For some reason, NVVL mangles the right-most column of pixels.
    # This is a known issue: https://github.com/NVIDIA/nvvl/issues/37
    # So, for our tests, we will simply ignore the last pixel column.
    a = np.array(actual_img)[:, :-1, :]
    b = np.array(expected_img)[:, :-1, :]
    np.testing.assert_allclose(a, b, rtol=0, atol=50)


def test_nvvl_read_frame(video_filename, first_frame_image):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)
    rgb = backend.read_frame()

    assert(rgb.size() == (3, 720, 1280))

    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')

    _assert_same_image(img, first_frame_image)


def test_nvvl_seek_to_frame(video_filename, mid_frame_image):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)
    backend.seek_to_frame(25)
    rgb = backend.read_frame()
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    _assert_same_image(img, mid_frame_image)


def test_nvvl_eof(video_filename):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)
    backend.seek(2.0)
    with pytest.raises(EOFError):
        backend.read_frame()


def test_nvvl_read_all_frames(video_filename):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)

    n_read = 0
    for i in range(1000):
        try:
            backend.read_frame()
            n_read += 1
        except EOFError:
            break
    assert n_read == 50


def test_nvvl_seek(video_filename, mid_frame_image):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)
    backend.seek(1.0)
    rgb = backend.read_frame()
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    _assert_same_image(img, mid_frame_image)


def test_nvvl_duration(video_filename):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)
    assert backend.duration == 2.0


def test_nvvl_frame_rate(video_filename):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)
    assert backend.frame_rate == 25


def test_nvvl_n_frames(video_filename):
    backend = NvvlBackendFactory().create(video_filename, 'cuda:0', torch.float32)
    assert backend.n_frames == 50
