from typing import Dict

from .backends import PyAvBackend, NvdecBackend
from .backends.common import Backend

device_backends: Dict[str, Backend] = {}


def set_device_backend(device_type, backend):
    device_backends[device_type] = backend


class VideoLoader:
    def __init__(self, filename, device):
        self.backend_inst = device_backends[device.type].create(filename, device)

    def seek(self, time_secs):
        self.backend_inst.seek(time_secs)

    def read_frame_rgb(self):
        return self.backend_inst.read_frame_rgb()


def _init():
    set_device_backend('cpu', PyAvBackend())
    set_device_backend('cuda', NvdecBackend())


_init()
