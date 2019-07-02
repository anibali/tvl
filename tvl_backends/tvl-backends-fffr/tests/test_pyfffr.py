import PIL.Image
import pyfffr
import pytest
import torch
from numpy.testing import assert_allclose

from tvl_backends.fffr.memory import TorchImageAllocator


def test_duration(video_filename):
    allocator = TorchImageAllocator('cpu', torch.uint8)
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, -1)
    assert fr.get_duration() == 2.0


def test_frame_rate(video_filename):
    allocator = TorchImageAllocator('cpu', torch.uint8)
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, -1)
    assert fr.get_frame_rate() == 25


def test_n_frames(video_filename):
    allocator = TorchImageAllocator('cpu', torch.uint8)
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, -1)
    assert fr.get_number_of_frames() == 50


def test_get_width(video_filename):
    allocator = TorchImageAllocator('cpu', torch.uint8)
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, -1)
    assert fr.get_width() == 1280


def test_get_height(video_filename):
    allocator = TorchImageAllocator('cpu', torch.uint8)
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, -1)
    assert fr.get_height() == 720


def test_seek_out_of_bounds(video_filename):
    allocator = TorchImageAllocator('cpu', torch.uint8)
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, -1)
    # Try seeking to a time after the end of the video.
    with pytest.raises(RuntimeError):
        fr.seek(2.0)
    # Try seeking to a time before the start of the video.
    with pytest.raises(RuntimeError):
        fr.seek(-1.0)
    # Check that seeking within bounds is still OK.
    fr.seek(1.0)


def test_two_decoders(video_filename):
    allocator = TorchImageAllocator('cuda:0', torch.uint8)
    fr1 = pyfffr.TvFFFrameReader(allocator, video_filename, allocator.device.index)
    fr1.read_frame()
    fr2 = pyfffr.TvFFFrameReader(allocator, video_filename, allocator.device.index)
    fr2.read_frame()


def test_device_works_after_reading_frame(device, video_filename):
    allocator = TorchImageAllocator(device, torch.uint8)
    cuda_tensor = torch.zeros(1, device=allocator.device)
    assert cuda_tensor.item() == 0
    gpu_index = allocator.device.index if allocator.device.type == 'cuda' else -1
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, gpu_index)
    fr.read_frame()
    assert cuda_tensor.item() == 0


def test_read_frame_uint8(device, video_filename, first_frame_image):
    allocator = TorchImageAllocator(device, torch.uint8)
    gpu_index = allocator.device.index if allocator.device.type == 'cuda' else -1
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, gpu_index)
    ptr = fr.read_frame()
    assert ptr is not None
    rgb_frame = allocator.get_frame_tensor(int(ptr))
    assert rgb_frame.shape == (3, 720, 1280)
    actual = PIL.Image.fromarray(rgb_frame.cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, first_frame_image, atol=50)


def test_read_frame_float32(device, video_filename, first_frame_image):
    if device == 'cpu':
        pytest.skip('Software decoding does not support float32 output directly')

    allocator = TorchImageAllocator(device, torch.float32)
    gpu_index = allocator.device.index if allocator.device.type == 'cuda' else -1
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, gpu_index)
    ptr = fr.read_frame()
    assert ptr is not None
    rgb_frame = allocator.get_frame_tensor(int(ptr))
    assert rgb_frame.shape == (3, 720, 1280)
    actual = PIL.Image.fromarray((rgb_frame.cpu() * 255).byte().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, first_frame_image, atol=50)


def test_read_frame_cropped(device, cropped_video_filename, cropped_first_frame_image):
    allocator = TorchImageAllocator(device, torch.uint8)
    gpu_index = allocator.device.index if allocator.device.type == 'cuda' else -1
    fr = pyfffr.TvFFFrameReader(allocator, cropped_video_filename, gpu_index)
    ptr = fr.read_frame()
    assert ptr is not None
    rgb_frame = allocator.get_frame_tensor(int(ptr))
    assert rgb_frame.shape == (3, 28, 52)
    actual = PIL.Image.fromarray(rgb_frame.cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, cropped_first_frame_image, atol=50)


def test_read_frames_by_index(device, video_filename, mid_frame_image):
    allocator = TorchImageAllocator(device, torch.uint8)
    gpu_index = allocator.device.index if allocator.device.type == 'cuda' else -1
    fr = pyfffr.TvFFFrameReader(allocator, video_filename, gpu_index)
    fr.seek(0.4)
    frame_offsets = torch.tensor([0, 10, 25, 30], device='cpu', dtype=torch.int64)
    ptrs = torch.zeros(frame_offsets.shape, device='cpu', dtype=torch.int64)
    res = fr.read_frames_by_index(frame_offsets.data_ptr(), len(frame_offsets), ptrs.data_ptr())
    assert res == len(frame_offsets)
    rgb_frame = allocator.get_frame_tensor(int(ptrs[2]))
    assert rgb_frame.shape == (3, 720, 1280)
    actual = PIL.Image.fromarray(rgb_frame.cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, mid_frame_image, atol=50)
