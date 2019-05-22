import torch

import pyfffr
from tvl.backend import Backend, BackendFactory
from tvl_backends.fffr.memory import TorchMemManager


class FffrBackend(Backend):
    def __init__(self, filename, device, dtype):
        device = torch.device(device)
        mem_manager = TorchMemManager(device)
        mem_manager.__disown__()
        self.mem_manager = mem_manager
        if device.type == 'cuda':
            device_index = device.index
        else:
            device_index = -1
        self.frame_reader = pyfffr.TvFFFrameReader(mem_manager, filename, device_index)
        self.dtype = dtype

    @property
    def duration(self):
        return self.frame_reader.get_duration()

    @property
    def frame_rate(self):
        return self.frame_reader.get_frame_rate()

    @property
    def n_frames(self):
        return self.frame_reader.get_number_of_frames()

    @property
    def width(self):
        return self.frame_reader.get_width()

    @property
    def height(self):
        return self.frame_reader.get_height()

    def seek(self, time_secs):
        self.frame_reader.seek(time_secs)

    def read_frame(self):
        ptr = self.frame_reader.read_frame()
        if ptr == 0:
            raise EOFError()
        # TODO: Actually return the frame in a usable form
        return None


class FffrBackendFactory(BackendFactory):
    def create(self, filename, device, dtype, backend_opts=None) -> FffrBackend:
        if backend_opts is None:
            backend_opts = {}
        return FffrBackend(filename, device, dtype, **backend_opts)
