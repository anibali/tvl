import os
from collections import namedtuple

import pytest
import pytest_mock
import torch

import tvl

mocker = pytest_mock.mocker
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# Get the slow CUDA initialisation out of the way
for i in range(torch.cuda.device_count()):
    torch.empty(0).to(torch.device('cuda', i))


@pytest.fixture
def dummy_backend():
    from tvl.backends.common import BackendInstance, Backend

    class DummyBackendInstance(BackendInstance):
        def __init__(self, frames):
            self.frames = frames
            self.pos = 0

        @property
        def duration(self):
            return len(self.frames) / self.frame_rate

        @property
        def frame_rate(self):
            return 10

        @property
        def n_frames(self):
            return int(self.duration * self.frame_rate)

        def read_frame(self):
            if self.pos < len(self.frames):
                frame = self.frames[self.pos]
                self.pos += 1
                return frame
            raise EOFError()

        def seek(self, *args):
            pass

    class DummyBackend(Backend):
        def __init__(self, frames):
            self.frames = frames
            self.device = namedtuple('DummyDevice', ['type'])('dummy')

        def create(self, *args):
            return DummyBackendInstance(self.frames)

    dummy_backend = DummyBackend([object() for _ in range(5)])
    tvl.set_device_backend(dummy_backend.device.type, dummy_backend)
    yield dummy_backend
    del tvl._device_backends[dummy_backend.device.type]
