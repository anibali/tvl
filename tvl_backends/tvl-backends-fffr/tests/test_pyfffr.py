import PIL.Image
import pyfffr
import pytest
import torch
from numpy.testing import assert_allclose

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


@pytest.mark.skip('This test currently crashes with SIGABRT.')
def test_seek_eof(video_filename):
    fr = pyfffr.TvFFFrameReader(None, video_filename, -1)
    fr.seek(2.0)


@pytest.mark.skip('This test currently crashes with SIGABRT.')
def test_two_decoders(video_filename):
    mm = TorchMemManager('cuda:0')
    fr1 = pyfffr.TvFFFrameReader(mm, video_filename, mm.device.index)
    fr1.read_frame()
    fr2 = pyfffr.TvFFFrameReader(mm, video_filename, mm.device.index)
    fr2.read_frame()


def test_device_works_after_reading_frame(device, video_filename):
    mm = TorchMemManager(device)
    cuda_tensor = torch.zeros(1, device=mm.device)
    assert cuda_tensor.item() == 0
    gpu_index = mm.device.index if mm.device.type == 'cuda' else -1
    fr = pyfffr.TvFFFrameReader(mm, video_filename, gpu_index)
    fr.read_frame()
    assert cuda_tensor.item() == 0


def test_read_frame(device, video_filename, first_frame_image):
    mm = TorchMemManager(device)
    gpu_index = mm.device.index if mm.device.type == 'cuda' else -1
    fr = pyfffr.TvFFFrameReader(mm, video_filename, gpu_index)
    ptr = fr.read_frame()
    assert ptr is not None
    # We expect for some memory to have been allocated via the MemManager
    assert len(mm.chunks) > 0
    expected_frame_size = 3 * 1280 * 720
    assert len(mm.find_containing_chunk(int(ptr))) >= expected_frame_size
    rgb_frame = mm.tensor(int(ptr), expected_frame_size)
    rgb_frame = rgb_frame.view(3, fr.get_height(), fr.get_width())
    rgb_frame = rgb_frame[[1, 2, 0], ...]
    actual = PIL.Image.fromarray(rgb_frame.cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, first_frame_image, atol=50)
