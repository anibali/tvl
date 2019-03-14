import PIL.Image
import numpy as np
import pytest


def _assert_same_image(actual_img, expected_img):
    # For some reason, NVVL mangles the right-most column of pixels.
    # This is a known issue: https://github.com/NVIDIA/nvvl/issues/37
    # So, for our tests, we will simply ignore the last pixel column.
    a = np.array(actual_img)[:, :-1, :]
    b = np.array(expected_img)[:, :-1, :]
    np.testing.assert_allclose(a, b, rtol=0, atol=50)


def test_read_frame(backend, first_frame_image):
    rgb = backend.read_frame()
    assert(rgb.size() == (3, 720, 1280))
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    _assert_same_image(img, first_frame_image)


def test_seek_to_frame(backend, mid_frame_image):
    backend.seek_to_frame(25)
    rgb = backend.read_frame()
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    _assert_same_image(img, mid_frame_image)


def test_eof(backend):
    backend.seek(2.0)
    with pytest.raises(EOFError):
        backend.read_frame()


def test_read_all_frames(backend):
    n_read = 0
    for i in range(1000):
        try:
            backend.read_frame()
            n_read += 1
        except EOFError:
            break
    assert n_read == 50


def test_seek(backend, mid_frame_image):
    backend.seek(1.0)
    rgb = backend.read_frame()
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    _assert_same_image(img, mid_frame_image)


def test_duration(backend):
    assert backend.duration == 2.0


def test_frame_rate(backend):
    assert backend.frame_rate == 25


def test_n_frames(backend):
    assert backend.n_frames == 50


def test_width(backend):
    assert backend.width == 1280


def test_height(backend):
    assert backend.height == 720
