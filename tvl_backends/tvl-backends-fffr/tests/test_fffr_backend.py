import PIL.Image
import numpy as np


def test_cpu_read_frame(cpu_backend, first_frame_image):
    rgb = cpu_backend.read_frame()
    assert(rgb.size() == (3, 720, 1280))
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    np.testing.assert_allclose(img, first_frame_image, rtol=0, atol=50)


# def test_eof(backend):
#     backend.seek(2.0)
#     with pytest.raises(EOFError):
#         backend.read_frame()


def test_read_all_frames(backend):
    n_read = 0
    for i in range(1000):
        try:
            backend.read_frame()
            n_read += 1
        except EOFError:
            break
    assert n_read == 50


def test_cpu_read_all_frames(cpu_backend):
    n_read = 0
    for i in range(1000):
        try:
            cpu_backend.read_frame()
            n_read += 1
        except EOFError:
            break
    assert n_read == 50


def test_cpu_seek(cpu_backend, mid_frame_image):
    cpu_backend.seek(1.0)
    rgb = cpu_backend.read_frame()
    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
    np.testing.assert_allclose(img, mid_frame_image, rtol=0, atol=50)


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