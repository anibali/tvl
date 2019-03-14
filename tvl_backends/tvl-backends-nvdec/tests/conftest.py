from pathlib import Path

import PIL.Image
import pytest
import torch

from tvl_backends.nvdec import NvdecBackendFactory

data_dir = Path(__file__).parent.parent.parent.parent.joinpath('data')

# Get the slow CUDA initialisation out of the way
for i in range(torch.cuda.device_count()):
    torch.empty(0).to(torch.device('cuda', i))


@pytest.fixture
def video_filename():
    return str(data_dir.joinpath('board_game-h264.mkv'))


@pytest.fixture
def first_frame_image():
    return PIL.Image.open(data_dir.joinpath('board_game_first.jpg'), 'r')


@pytest.fixture
def mid_frame_image():
    return PIL.Image.open(data_dir.joinpath('board_game_mid.jpg'), 'r')


@pytest.fixture
def backend(video_filename):
    return NvdecBackendFactory().create(video_filename, 'cuda:0', torch.float32)
