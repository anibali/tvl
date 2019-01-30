from typing import Dict
from warnings import warn

import torch

import tvl.backends
from tvl.backends.common import Backend

_backend_priorities = {
    'cpu': ['PyAvBackend', 'OpenCvBackend'],
    'cuda': ['NvdecBackend'],
}
_device_backends: Dict[str, Backend] = {}


def set_device_backend(device_type, backend):
    _device_backends[device_type] = backend


class VideoLoader:
    def __init__(self, filename, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.backend_inst = _device_backends[device.type].create(filename, device)

    def seek(self, time_secs):
        self.backend_inst.seek(time_secs)

    def seek_to_frame(self, frame_index):
        self.seek(frame_index / self.backend_inst.frame_rate)

    def read_frame(self):
        return self.backend_inst.read_frame()

    def read_frames(self):
        more_frames = True
        while more_frames:
            try:
                yield self.read_frame()
            except EOFError:
                more_frames = False

    @property
    def duration(self):
        return self.backend_inst.duration

    @property
    def frame_rate(self):
        return self.backend_inst.frame_rate

    @property
    def n_frames(self):
        return self.backend_inst.n_frames

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


def _init():
    unsupported_devices = []
    for device_type, backend_names in _backend_priorities.items():
        for backend_name in backend_names:
            if hasattr(tvl.backends, backend_name):
                set_device_backend(device_type, getattr(tvl.backends, backend_name)())
                break
        else:
            unsupported_devices.append(device_type)
    if len(unsupported_devices) > 0:
        warn(f'No video loading backend available for the following devices: {repr(unsupported_devices)}')


_init()
