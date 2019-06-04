import pyfffr
import pytest
from tvl_backends.fffr.memory import TorchMemManager
import PIL.Image
from numpy.testing import assert_allclose


def test_duration(video_filename):
    fr = pyfffr.TvFFFrameReader(None, video_filename, -1)
    assert fr.get_duration() == 2.0


def test_frame_rate(video_filename):
    fr = pyfffr.TvFFFrameReader(None, video_filename, -1)
    assert fr.get_frame_rate() == 25


def test_n_frames(video_filename):
    fr = pyfffr.TvFFFrameReader(None, video_filename, -1)
    assert fr.get_number_of_frames() == 50


def test_get_width(video_filename):
    fr = pyfffr.TvFFFrameReader(None, video_filename, -1)
    assert fr.get_width() == 1280


def test_get_height(video_filename):
    fr = pyfffr.TvFFFrameReader(None, video_filename, -1)
    assert fr.get_height() == 720


@pytest.mark.skip('This test currently crashes with SIGABRT.')
def test_seek_eof(video_filename):
    fr = pyfffr.TvFFFrameReader(None, video_filename, -1)
    fr.seek(2.0)


@pytest.mark.skip('This test currently crashes with SIGSEGV.')
def test_two_decoders(video_filename):
    fr1 = pyfffr.TvFFFrameReader(None, video_filename, 0)
    fr2 = pyfffr.TvFFFrameReader(None, video_filename, 0)
    fr1.read_frame()
    fr2.read_frame()


def test_cpu_read_frame(video_filename, first_frame_image):
    mm = TorchMemManager('cpu')
    fr = pyfffr.TvFFFrameReader(mm, video_filename, -1)
    ptr = fr.read_frame()
    assert ptr is not None
    # We expect for some memory to have been allocated via the MemManager
    assert len(mm.chunks) > 0
    rgb_frame = mm.tensor(int(ptr), fr.get_frame_size())
    rgb_frame = rgb_frame.view(3, fr.get_height(), fr.get_width())
    rgb_frame = rgb_frame[[1, 2, 0], ...]
    actual = PIL.Image.fromarray(rgb_frame.cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, first_frame_image, atol=50)


@pytest.mark.skip('This test currently crashes with SIGSEGV.')
def test_gpu_read_frame(video_filename, first_frame_image):
    mm = TorchMemManager('cuda:0')
    fr = pyfffr.TvFFFrameReader(mm, video_filename, -1)
    ptr = fr.read_frame()
    assert ptr is not None
    # We expect for some memory to have been allocated via the MemManager
    assert len(mm.chunks) > 0
    rgb_frame = mm.tensor(int(ptr), fr.get_frame_size())
    rgb_frame = rgb_frame.view(3, fr.get_height(), fr.get_width())
    rgb_frame = rgb_frame[[1, 2, 0], ...]
    actual = PIL.Image.fromarray(rgb_frame.cpu().permute(1, 2, 0).numpy(), 'RGB')
    actual.save('something.png')
    assert_allclose(actual, first_frame_image, atol=50)
