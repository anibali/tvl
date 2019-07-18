import os
import time

import numpy as np
import torch

import tvl
from tvl_backends.fffr import FffrBackendFactory
from tvl_backends.nvdec import NvdecBackendFactory
from tvl_backends.pyav import PyAvBackendFactory

video_file = os.path.join(os.path.dirname(__file__), '../data/board_game-h264.mkv')


def read_sequential(video_file, device, **backend_opts):
    n_frames = 40
    n_trials = 1

    vl = tvl.VideoLoader(video_file, device, torch.float32, backend_opts)

    # Read one frame to get any initialisation out of the way.
    vl.read_frame()

    t1 = time.time()
    for _ in range(n_trials):
        frames = list(vl.read_frames(n_frames))
        assert len(frames) == n_frames
    t2 = time.time()

    return (n_trials * n_frames) / (t2 - t1)


def read_random(video_file, device, **backend_opts):
    n_frames = 5
    n_trials = 8

    vl = tvl.VideoLoader(video_file, device, torch.float32, backend_opts)

    # Read one frame to get any initialisation out of the way.
    vl.read_frame()
    vl.seek(0)

    t1 = time.time()
    for _ in range(n_trials):
        frames = list(vl.select_frames(np.arange(n_frames) * 10))
        assert len(frames) == n_frames
    t2 = time.time()

    return (n_trials * n_frames) / (t2 - t1)


def main():
    # Get the slow CUDA initialisation out of the way.
    for i in range(torch.cuda.device_count()):
        torch.empty(0).to(torch.device('cuda', i))

    backends = [
        ('nvdec-cuda', NvdecBackendFactory, 'cuda'),
        ('fffr-cuda', FffrBackendFactory, 'cuda'),
        ('pyav-cpu', PyAvBackendFactory, 'cpu'),
        ('fffr-cpu', FffrBackendFactory, 'cpu'),
    ]

    print('+++ SEQUENTIAL +++')
    for name, factory_cls, device_type in backends:
        tvl.set_backend_factory(device_type, factory_cls())
        fps = read_sequential(video_file, device_type)
        print(f'{name:12s} {fps:10.2f}')

    print()

    print('+++ RANDOM +++')
    for name, factory_cls, device_type in backends:
        tvl.set_backend_factory(device_type, factory_cls())
        fps = read_random(video_file, device_type, seek_threshold=5)
        print(f'{name:12s} {fps:10.2f}')


if __name__ == '__main__':
    main()
