import PIL.Image
import numpy as np
import pytest

from tvl.backends import PyAvBackend


def test_pyav_read_frame(video_filename, first_frame_image):
    backend = PyAvBackend()

    inst = backend.create(video_filename, 'cpu')
    rgb = inst.read_frame()

    assert(rgb.size() == (3, 720, 1280))

    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')

    np.testing.assert_allclose(img, first_frame_image, rtol=0, atol=50)


def test_pyav_eof(video_filename):
    backend = PyAvBackend()

    inst = backend.create(video_filename, 'cpu')
    inst.seek(2.0)
    with pytest.raises(EOFError):
        inst.read_frame()


def test_pyav_seek(video_filename, mid_frame_image):
    backend = PyAvBackend()

    inst = backend.create(video_filename, 'cpu')
    inst.seek(1.0)
    rgb = inst.read_frame()
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    np.testing.assert_allclose(img, mid_frame_image, rtol=0, atol=50)


def test_pyav_duration(video_filename):
    backend = PyAvBackend()
    inst = backend.create(video_filename, 'cpu')
    assert inst.duration == 2.0


def test_pyav_frame_rate(video_filename):
    backend = PyAvBackend()
    inst = backend.create(video_filename, 'cpu')
    assert inst.frame_rate == 25
