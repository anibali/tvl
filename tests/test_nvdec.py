import PIL.Image
import numpy as np
import pytest

from tvl.backends import NvdecBackend


def test_nvdec_read_frame(video_filename, first_frame_image):
    backend = NvdecBackend()

    inst = backend.create(video_filename, 'cuda:0')
    rgb = inst.read_frame()

    assert(rgb.size() == (3, 720, 1280))

    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')

    np.testing.assert_allclose(img, first_frame_image, rtol=0, atol=50)


def test_nvdec_eof(video_filename):
    backend = NvdecBackend()

    inst = backend.create(video_filename, 'cuda:0')
    inst.seek(2.0)
    with pytest.raises(EOFError):
        inst.read_frame()


def test_nvdec_seek(video_filename, mid_frame_image):
    backend = NvdecBackend()

    inst = backend.create(video_filename, 'cuda:0')
    inst.seek(1.0)
    rgb = inst.read_frame()
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    np.testing.assert_allclose(img, mid_frame_image, rtol=0, atol=50)
