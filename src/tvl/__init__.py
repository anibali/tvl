from typing import Dict

from .backends import PyAvBackend, NvdecBackend
from .backends.common import Backend

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


def _init():
    set_device_backend('cpu', PyAvBackend())
    set_device_backend('cuda', NvdecBackend())


_init()
