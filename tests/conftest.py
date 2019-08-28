from pathlib import Path

import pytest
import pytest_mock
import torch

import tvl
from tvl.backend import Backend, BackendFactory

mocker = pytest_mock.mocker
data_dir = Path(__file__).parent.parent.joinpath('data')

# Get the slow CUDA initialisation out of the way
for i in range(torch.cuda.device_count()):
    torch.empty(0).to(torch.device('cuda', i))


class DummyBackend(Backend):
    def __init__(self, frames, video_filename, device):
        super().__init__(video_filename, device, torch.float32, 3)
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
    def __init__(self, frames, video_filename, device):
        self.frames = frames
        self.video_filename = video_filename
        self.device = device

    def create(self, *args):
        return DummyBackend(self.frames, self.video_filename, self.device)


@pytest.fixture()
def dummy_backend_factory_cpu(video_filename):
    frames = [object() for _ in range(5)]
    return DummyBackendFactory(frames, video_filename, 'cpu')


@pytest.fixture()
def dummy_backend_factory_cuda(video_filename):
    frames = [object() for _ in range(5)]
    return DummyBackendFactory(frames, video_filename, 'cuda')


@pytest.fixture(autouse=True)
def set_dummy_backend_factories(dummy_backend_factory_cpu, dummy_backend_factory_cuda):
    old_device_backends = tvl._device_backends
    tvl.set_backend_factory('cpu', dummy_backend_factory_cpu)
    tvl.set_backend_factory('cuda', dummy_backend_factory_cuda)
    yield
    tvl._device_backends = old_device_backends


@pytest.fixture
def video_filename():
    return str(data_dir.joinpath('board_game-h264.mkv'))
