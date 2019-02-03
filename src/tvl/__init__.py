import importlib
from typing import Dict

import torch

import tvl.backend
from tvl.backend import BackendFactory

# Explicitly set backends for particular device types.
_device_backends: Dict[str, BackendFactory] = {}
# Known backends. These will be searched if a device type does not have a backend factory
# set explicitly.
_known_backends = {
    'cpu': [
        'tvl_backends.pyav.PyAvBackendFactory',     # PyPI package: tvl-backends-pyav
        'tvl_backends.opencv.OpenCvBackendFactory', # PyPI package: tvl-backends-opencv
    ],
    'cuda': [
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
    def __init__(self, filename, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.backend = get_backend_factory(device.type).create(filename, device)

    def seek(self, time_secs):
        self.backend.seek(time_secs)

    def seek_to_frame(self, frame_index):
        self.seek(frame_index / self.backend.frame_rate)

    def read_frame(self):
        return self.backend.read_frame()

    def read_frames(self):
        more_frames = True
        while more_frames:
            try:
                yield self.read_frame()
            except EOFError:
                more_frames = False

    @property
    def duration(self):
        return self.backend.duration

    @property
    def frame_rate(self):
        return self.backend.frame_rate

    @property
    def n_frames(self):
        return self.backend.n_frames

    def pick_frames(self, frame_indices, skip_threshold=3):
        """

        Args:
            frame_indices (list of int): Indices of frames to read.
            skip_threshold (int, optional): Sequential reading threshold used to predict when
                multiple reads will be faster than seeking. Setting this value close to the video's
                GOP size should be a reasonable choice.

        Returns:
            list of torch.Tensor: RGB frames corresponding to `frame_indices`.
        """
        # We will be loading unique frames in ascending index order.
        sorted_frame_indices = list(sorted(set(frame_indices)))
        frames = {}

        pos = -(skip_threshold + 1)
        for frame_index in sorted_frame_indices:
            if frame_index - pos > skip_threshold:
                # Skip to desired location by seeking.
                self.seek_to_frame(frame_index)
            else:
                # Skip to desired location by reading and discarding intermediate frames.
                while pos < frame_index:
                    self.read_frame()
                    pos += 1
            # Read the frame that we care about.
            frames[frame_index] = self.read_frame()
            pos = frame_index + 1

        # Order frames to correspond with `frame_indices`.
        return [frames[frame_index] for frame_index in frame_indices]
