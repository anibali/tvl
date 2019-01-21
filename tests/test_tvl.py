import os
import PIL.Image
import numpy as np
from collections import namedtuple

import tvl
from tvl.backends import NvdecBackend, PyAvBackend
from tvl.backends.common import BackendInstance, Backend

data_dir = os.path.join(os.path.dirname(__file__), 'data')
VIDEO_FILENAME = os.path.join(data_dir, 'board_game.mkv')
FIRST_FRAME = PIL.Image.open(os.path.join(data_dir, 'board_game_first.jpg'), 'r')


def test_nvdec_backend():
    backend = NvdecBackend()

    inst = backend.create(VIDEO_FILENAME, 'cuda:0')
    rgb = inst.read_frame_rgb()

    assert(rgb.size() == (3, 720, 1280))

    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')

    np.testing.assert_allclose(img, FIRST_FRAME, rtol=0, atol=50)


def test_pyav_backend():
    backend = PyAvBackend()

    inst = backend.create(VIDEO_FILENAME, 'cpu')
    rgb = inst.read_frame_rgb()

    assert(rgb.size() == (3, 720, 1280))

    rgb_bytes = (rgb * 255).round_().byte().cpu()
    img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')

    np.testing.assert_allclose(img, FIRST_FRAME, rtol=0, atol=50)


def test_set_device_backend():
    class DummyBackendInstance(BackendInstance):
        def read_frame_rgb(self):
            return 'dummy_frame'
        def seek(self, *args):
            pass
    class DummyBackend(Backend):
        def create(self, *args):
            return DummyBackendInstance()
    device = namedtuple('DummyDevice', ['type'])('dummy')
    tvl.set_device_backend(device.type, DummyBackend())
    vl = tvl.VideoLoader(VIDEO_FILENAME, device)
    assert(vl.read_frame_rgb() == 'dummy_frame')
