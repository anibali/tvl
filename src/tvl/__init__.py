import importlib
from typing import Dict, Sequence

import torch

import tvl.backend
from tvl.backend import BackendFactory

# Explicitly set backends for particular device types.
_device_backends: Dict[str, BackendFactory] = {}
# Known backends. These will be searched if a device type does not have a backend factory
# set explicitly.
_known_backends = {
    'cpu': [
        'tvl_backends.fffr.FffrBackendFactory',     # PyPI package: tvl-backends-fffr
        'tvl_backends.pyav.PyAvBackendFactory',     # PyPI package: tvl-backends-pyav
        'tvl_backends.opencv.OpenCvBackendFactory', # PyPI package: tvl-backends-opencv
    ],
    'cuda': [
        'tvl_backends.fffr.FffrBackendFactory',     # PyPI package: tvl-backends-fffr
        'tvl_backends.nvdec.NvdecBackendFactory',   # PyPI package: tvl-backends-nvdec
    ],
}


def set_backend_factory(device_type, backend_factory):
    """Set the backend factory to be used for a particular device type."""
    _device_backends[device_type] = backend_factory


def _auto_set_backend_factory(device_type):
    """Attempt to automatically set the backend for `device_type` if not set already."""
    if device_type in _device_backends and _device_backends[device_type] is not None:
        return
    if device_type in _known_backends:
        for backend_name in _known_backends[device_type]:
            try:
                module_name, class_name = backend_name.rsplit('.', 1)
                module = importlib.import_module(module_name)
                set_backend_factory(device_type, getattr(module, class_name)())
                return
            except ImportError:
                pass


def get_backend_factory(device_type) -> BackendFactory:
    """Get the backend factory which will be used for a particular device type."""
    _auto_set_backend_factory(device_type)
    if device_type in _device_backends:
        return _device_backends[device_type]
    raise Exception(f'failed to find a backend factory for device type: {device_type}')


class VideoLoader:
    def __init__(self, filename, device, dtype=torch.float32, backend_opts=None):
        if isinstance(device, str):
            device = torch.device(device)
        self.backend = get_backend_factory(device.type).create(filename, device, dtype, backend_opts)

    def seek(self, time_secs):
        self.backend.seek(time_secs)

    def seek_to_frame(self, frame_index):
        self.backend.seek_to_frame(frame_index)

    def read_frame(self):
        return self.backend.read_frame()

    def read_frames(self, n):
        return self.backend.read_frames(n)

    @property
    def duration(self):
        return self.backend.duration

    @property
    def frame_rate(self):
        return self.backend.frame_rate

    @property
    def n_frames(self):
        return self.backend.n_frames

    @property
    def width(self):
        return self.backend.width

    @property
    def height(self):
        return self.backend.height

    def remaining_frames(self):
        """Iterate sequentially over remaining frames in the video."""
        more_frames = True
        while more_frames:
            try:
                yield self.read_frame()
            except EOFError:
                more_frames = False

    def read_all_frames(self):
        """Iterate over all frames in the video."""
        self.seek_to_frame(0)
        return self.remaining_frames()

    def select_frames(self, frame_indices):
        """Iterate over frames selected by frame index.

        Frames will be yielded in ascending order of frame index, regardless of the way
        `frame_indices` is ordered.

        Args:
            frame_indices (Sequence of int): Indices of frames to read.
        """
        return self.backend.select_frames(frame_indices)
