import os
from collections import namedtuple

import PIL.Image
import pytest

import tvl

data_dir = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def video_filename():
    return os.path.join(data_dir, 'board_game.mkv')


@pytest.fixture
def first_frame_image():
    return PIL.Image.open(os.path.join(data_dir, 'board_game_first.jpg'), 'r')


@pytest.fixture
def mid_frame_image():
    return PIL.Image.open(os.path.join(data_dir, 'board_game_mid.jpg'), 'r')


@pytest.fixture
def dummy_backend():
    from tvl.backends.common import BackendInstance, Backend

    class DummyBackendInstance(BackendInstance):
        def __init__(self, frames):
            self.frames = frames
            self.pos = 0

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
