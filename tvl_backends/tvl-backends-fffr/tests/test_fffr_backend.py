import PIL.Image
import pytest
import torch
from numpy.testing import assert_allclose

from tvl_backends.fffr import FffrBackendFactory


def test_memory_leakage(backend):
    """Check that the memory manager is not leaking memory."""
    device = backend.image_allocator.device
    if device.type == 'cuda':
        start_mem = torch.cuda.memory_allocated(device.index)
        for _ in range(5):
            backend.read_frame()
        end_mem = torch.cuda.memory_allocated(device.index)
        assert end_mem == start_mem
    else:
        for _ in range(5):
            backend.read_frame()
    assert len(backend.image_allocator.tensors) == 0


def test_read_frame_float32_cpu(video_filename, first_frame_image):
    backend = FffrBackendFactory().create(video_filename, 'cpu', torch.float32)
    rgb_frame = backend.read_frame()
    assert rgb_frame.shape == (3, 720, 1280)
    actual = PIL.Image.fromarray((rgb_frame * 255).byte().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, first_frame_image, atol=50)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available.')
def test_cuda_device_without_index(video_filename, first_frame_image):
    backend = FffrBackendFactory().create(video_filename, 'cuda', torch.uint8)
    rgb_frame = backend.read_frame()
    assert rgb_frame.shape == (3, 720, 1280)
    actual = PIL.Image.fromarray(rgb_frame.cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, first_frame_image, atol=50)


def test_select_frames(device, video_filename, first_frame_image, mid_frame_image):
    backend = FffrBackendFactory().create(video_filename, device, torch.uint8)
    frames = list(backend.select_frames([0, 25]))
    assert(len(frames) == 2)
    actual = PIL.Image.fromarray(frames[0].cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, first_frame_image, atol=50)
    actual = PIL.Image.fromarray(frames[1].cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, mid_frame_image, atol=50)


def test_select_many_frames(device, video_filename, first_frame_image, mid_frame_image):
    backend = FffrBackendFactory().create(video_filename, device, torch.uint8)
    frames = list(backend.select_frames(list(range(26))))
    assert(len(frames) == 26)
    actual = PIL.Image.fromarray(frames[0].cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, first_frame_image, atol=50)
    actual = PIL.Image.fromarray(frames[25].cpu().permute(1, 2, 0).numpy(), 'RGB')
    assert_allclose(actual, mid_frame_image, atol=50)
