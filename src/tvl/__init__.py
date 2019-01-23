from typing import Dict

import tvl.backends
from tvl.backends.common import Backend
from warnings import warn

_backend_priorities = {
    'cpu': ['PyAvBackend'],
    'cuda': ['NvdecBackend'],
}
_device_backends: Dict[str, Backend] = {}


def set_device_backend(device_type, backend):
    _device_backends[device_type] = backend


class VideoLoader:
    def __init__(self, filename, device):
        self.backend_inst = _device_backends[device.type].create(filename, device)

    def seek(self, time_secs):
        self.backend_inst.seek(time_secs)

    def read_frame(self):
        return self.backend_inst.read_frame()

    def read_frames(self):
        more_frames = True
        while more_frames:
            try:
                yield self.read_frame()
            except EOFError:
                more_frames = False

    def pick_frames(self, frame_indices, stride_threshold=3):
        """

        Args:
            frame_indices (list of int): Indices of frames to read.
            stride_threshold (int, optional): Sequential reading threshold used to predict when
                multiple reads will be faster than seeking. Setting this value close to the video's
                GOP size should be a reasonable choice.

        Returns:
            list of torch.Tensor: RGB frames corresponding to `frame_indices`.
        """
        sorted_frame_indices = list(sorted(set(frame_indices)))
        first_frame = sorted_frame_indices[0]
        last_frame = sorted_frame_indices[-1]
        avg_frame_stride = (last_frame - first_frame) / len(sorted_frame_indices)

        frames = {}

        if avg_frame_stride > stride_threshold:
            # Read frames using random access
            for frame_index in sorted_frame_indices:
                self.seek(frame_index / self.backend_inst.frame_rate)
                frames[frame_index] = self.read_frame()
        else:
            # Read frames sequentially
            self.seek(first_frame / self.backend_inst.frame_rate)
            pos = first_frame
            for frame_index in sorted_frame_indices:
                while pos <= frame_index:
                    frames[frame_index] = self.read_frame()
                    pos += 1

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
