import os
import PIL.Image
import numpy as np

from tvl.backends import NvdecBackend

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
