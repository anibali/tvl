import importlib
import os
from contextlib import contextmanager
from threading import RLock, Condition
from typing import Dict, Sequence, Iterator, Union

import torch

import tvl.backend
from tvl.backend import BackendFactory

# Explicitly set backends for particular device types.
_device_backends: Dict[str, BackendFactory] = {}
# Known backends. These will be searched if a device type does not have a backend factory
# set explicitly.
_known_backends = {
    'cpu': [
        # 'tvl_backends.fffr.FffrBackendFactory',     # PyPI package: tvl-backends-fffr
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
    def __init__(self, filename, device: Union[torch.device, str], dtype=torch.float32, backend_opts=None):
        if isinstance(device, str):
            device = torch.device(device)
        filename = os.fspath(filename)
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
        `frame_indices` is ordered. Duplicate frame indices will be ignored.

        Args:
            frame_indices (Sequence of int): Indices of frames to read.

        Returns:
            Iterator[torch.Tensor]: An iterator of image tensors.
        """
        return self.backend.select_frames(frame_indices)

    def select_frame(self, frame_index):
        """Read a single frame by frame index.

        Args:
            frame_index (int): Index of frame to read.

        Returns:
            torch.Tensor: Frame image tensor.
        """
        return self.backend.select_frame(frame_index)


class VideoLoaderPool:
    def __init__(self, slots: Dict[str, int]):
        self.slots = slots
        self.condition = Condition(RLock())

    def peek_slot(self):
        for device, available in self.slots.items():
            if available > 0:
                return device
        return None

    def remove_slot(self):
        device = self.peek_slot()
        if device is None:
            raise Exception('No slots available')
        self.slots[device] -= 1
        return device

    def add_slots(self, device, n=1):
        available = self.slots.get(device, 0)
        self.slots[device] = available + n

    @contextmanager
    def loader(self, filename, dtype=torch.float32, backend_opts_by_device=None):
        with self.condition:
            while self.peek_slot() is None:
                self.condition.wait()
            device = self.remove_slot()

        if backend_opts_by_device is None:
            backend_opts_by_device = {}

        try:
            yield VideoLoader(filename, device, dtype, backend_opts_by_device.get(device, None))
        finally:
            with self.condition:
                self.add_slots(device, 1)
                self.condition.notify()
