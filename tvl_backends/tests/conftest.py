from pathlib import Path

import PIL.Image
import pytest
import torch

from tvl_backends.fffr import FffrBackendFactory
from tvl_backends.nvdec import NvdecBackendFactory
from tvl_backends.pyav import PyAvBackendFactory
from tvl_backends.opencv import OpenCvBackendFactory

FACTORY_CLASSES = {
    'fffr': FffrBackendFactory,
    'nvdec': NvdecBackendFactory,
    'pyav': PyAvBackendFactory,
    'opencv': OpenCvBackendFactory,
}

data_dir = Path(__file__).parent.parent.parent.joinpath('data')

# Get the slow CUDA initialisation out of the way
CUDA_DEVICES = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
for device in CUDA_DEVICES:
    torch.empty(0).to(device)


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
def cropped_video_filename():
    return str(data_dir.joinpath('board_game-h264-cropped.mkv'))


@pytest.fixture
def cropped_first_frame_image():
    return PIL.Image.open(data_dir.joinpath('board_game_first-cropped.jpg'), 'r')


@pytest.fixture
def swimming_video_filename():
    return str(data_dir.joinpath('swimming-h264.mp4'))


@pytest.fixture
def swimming_mid_image():
    return PIL.Image.open(data_dir.joinpath('swimming_mid.jpg'), 'r')


cpu_backends = [
    # 'fffr',  # This currently causes a freeze due to some kind of conflict with PyAV.
    'pyav',
    'opencv',
]
cuda_backends = [
    'fffr',
    'nvdec',
]

backend_params = [(k, 'cpu') for k in cpu_backends]
if torch.cuda.is_available():
    backend_params.extend((k, 'cuda:0') for k in cuda_backends)
backend_ids = ['-'.join(e) for e in backend_params]


@pytest.fixture(params=backend_params, ids=backend_ids)
def backend_factory_and_device(request):
    factory_key, device_str = request.param
    return FACTORY_CLASSES[factory_key](), torch.device(device_str)


@pytest.fixture(params=['uint8', 'float32'])
def dtype(request):
    return getattr(torch, request.param)


@pytest.fixture
def backend(backend_factory_and_device, dtype, video_filename):
    assert(Path(video_filename).is_file())
    backend_factory, device = backend_factory_and_device
    backend = backend_factory.create(video_filename, device, dtype)
    return backend


@pytest.fixture
def resizing_backend(backend_factory_and_device, dtype, video_filename):
    assert(Path(video_filename).is_file())
    backend_factory, device = backend_factory_and_device
    backend = backend_factory.create(video_filename, device, dtype, backend_opts=dict(out_width=160, out_height=90))
    return backend
