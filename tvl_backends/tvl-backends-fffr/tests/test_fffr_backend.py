from concurrent.futures import ThreadPoolExecutor

import PIL.Image
import pytest
import torch

from tvl.testing import assert_same_image
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
    assert_same_image(rgb_frame, first_frame_image)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available.')
def test_cuda_device_without_index(video_filename, first_frame_image):
    backend = FffrBackendFactory().create(video_filename, 'cuda', torch.uint8)
    rgb_frame = backend.read_frame()
    assert rgb_frame.shape == (3, 720, 1280)
    assert_same_image(rgb_frame, first_frame_image)


def test_out_size(device, video_filename, first_frame_image):
    backend = FffrBackendFactory().create(video_filename, device, torch.uint8,
                                          backend_opts=dict(out_width=1140, out_height=360))
    rgb_frame = backend.read_frame()
    assert rgb_frame.shape == (3, 360, 1140)
    expected = first_frame_image.resize((1140, 360), resample=PIL.Image.BILINEAR)
    assert_same_image(rgb_frame, expected)


def test_select_frames(device, video_filename, first_frame_image, mid_frame_image):
    backend = FffrBackendFactory().create(video_filename, device, torch.uint8)
    frames = list(backend.select_frames([0, 25]))
    assert(len(frames) == 2)
    assert_same_image(frames[0], first_frame_image)
    assert_same_image(frames[1], mid_frame_image)


def test_select_many_frames(device, video_filename, first_frame_image, mid_frame_image):
    backend = FffrBackendFactory().create(video_filename, device, torch.uint8)
    frames = list(backend.select_frames(list(range(26))))
    assert(len(frames) == 26)
    assert_same_image(frames[0], first_frame_image)
    assert_same_image(frames[25], mid_frame_image)


def test_select_frames_diving_video(device, diving_video_filename, diving_frame07_image):
    backend = FffrBackendFactory().create(diving_video_filename, device, torch.uint8)
    frames = list(backend.select_frames(list(range(16))))
    assert(len(frames) == 16)
    assert_same_image(frames[7], diving_frame07_image)


def test_multithreading(device, video_filename, first_frame_image, mid_frame_image):
    executor = ThreadPoolExecutor(max_workers=8)
    seq_len = 5
    backend = FffrBackendFactory().create(video_filename, device, torch.uint8)

    def get(index):
        frames = list(backend.select_frames(list(range(index * seq_len, (index + 1) * seq_len))))
        return frames

    jobs = [executor.submit(get, i) for i in range(8)]
    results = [job.result() for job in jobs]

    assert_same_image(results[0][0], first_frame_image)
    assert_same_image(results[25 // seq_len][25 % seq_len], mid_frame_image)


def test_swimming_video(device, swimming_video_filename, swimming_mid_image):
    backend_opts = {'buffer_length': 1}
    backend = FffrBackendFactory().create(swimming_video_filename, device, torch.uint8, backend_opts)
    frame = backend.select_frame(9)
    assert_same_image(frame, swimming_mid_image, allow_mismatch=0.001)


def test_swimming_video_resized(device, swimming_video_filename, swimming_mid_image):
    backend_opts = {'buffer_length': 1, 'out_width': 1280, 'out_height': 720}
    backend = FffrBackendFactory().create(swimming_video_filename, device, torch.uint8, backend_opts)
    frame = backend.select_frame(9)
    expected = swimming_mid_image.resize((backend_opts['out_width'], backend_opts['out_height']))
    assert_same_image(frame, expected, allow_mismatch=0.001)
