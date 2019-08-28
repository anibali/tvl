from collections import namedtuple
from pathlib import Path

import pytest
import pytest_mock
import torch

import tvl

mocker = pytest_mock.mocker
data_dir = Path(__file__).parent.parent.joinpath('data')

# Get the slow CUDA initialisation out of the way
for i in range(torch.cuda.device_count()):
    torch.empty(0).to(torch.device('cuda', i))


@pytest.fixture
def video_filename():
    return str(data_dir.joinpath('board_game-h264.mkv'))


@pytest.fixture
def dummy_backend(video_filename):
    from tvl.backend import Backend, BackendFactory

    class DummyBackend(Backend):
        def __init__(self, frames):
            super().__init__(video_filename, 'cpu', torch.float32, 3)
            self.frames = frames
            self.pos = 0

        @property
        def duration(self):
            return len(self.frames) / self.frame_rate

        @property
        def frame_rate(self):
            return 10

        @property
        def width(self):
            return 800

        @property
        def height(self):
            return 600

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

    class DummyBackendFactory(BackendFactory):
        def __init__(self, frames):
            self.frames = frames
            self.device = namedtuple('DummyDevice', ['type'])('dummy')

        def create(self, *args):
            return DummyBackend(self.frames)

    dummy_backend = DummyBackendFactory([object() for _ in range(5)])
    tvl.set_backend_factory(dummy_backend.device.type, dummy_backend)
    yield dummy_backend
    del tvl._device_backends[dummy_backend.device.type]
