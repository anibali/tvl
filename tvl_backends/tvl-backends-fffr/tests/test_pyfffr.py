import pyfffr
import pytest
from tvl_backends.fffr.memory import TorchMemManager


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


@pytest.mark.skip('This test currently fails, awaiting fix in FFFrameReader')
def test_seek_eof(video_filename):
    fr = pyfffr.TvFFFrameReader(None, video_filename, -1)
    fr.seek(2.0)


@pytest.mark.skip('This test currently fails (https://github.com/Sibras/FFFrameReader/issues/3)')
def test_two_decoders(video_filename):
    print()
    fr1 = pyfffr.TvFFFrameReader(None, video_filename, 0)
    fr2 = pyfffr.TvFFFrameReader(None, video_filename, 0)
    fr2.read_frame()


@pytest.mark.skip('This test is a work in progress.')
def test_gpu_read_frame(video_filename):
    mm = TorchMemManager('cuda:0')
    fr = pyfffr.TvFFFrameReader(mm, video_filename, 0)
    yuv422 = fr.read_frame()
    # We expect for some memory to have been allocated via the MemManager
    assert len(mm.chunks) > 0

    # TODO: Assert checking that the read frame is good
    # yuv422_ctype = ctypes.cast(yuv422, ctypes.POINTER(ctypes.c_size_t))
    # y = mm.find_containing_chunk(yuv422_ctype[0])
    # u = mm.find_containing_chunk(yuv422_ctype[1])
    # v = mm.find_containing_chunk(yuv422_ctype[2])
