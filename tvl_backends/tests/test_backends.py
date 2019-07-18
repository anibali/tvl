import PIL.Image
import pytest
import torch
from numpy.testing import assert_allclose


def assert_same_image(actual, expected, atol=50):
    if torch.is_tensor(actual):
        if actual.is_floating_point():
            actual = actual * 255
        actual = actual.to(device='cpu', dtype=torch.uint8)
        actual = PIL.Image.fromarray(actual.permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, expected, rtol=0, atol=atol)


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


def test_read_frame(backend, first_frame_image):
    rgb = backend.read_frame()
    assert(rgb.size() == (3, 720, 1280))
    assert_same_image(rgb, first_frame_image)


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
    assert_same_image(rgb, mid_frame_image)


def test_select_frames(backend, first_frame_image, mid_frame_image):
    frames = list(backend.select_frames([0, 25]))
    assert_same_image(frames[0], first_frame_image)
    assert_same_image(frames[1], mid_frame_image)
