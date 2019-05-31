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
        if not ptr:
            raise EOFError()

        brg = self.mem_manager.tensor(int(ptr), self.frame_reader.get_frame_size())
        brg = brg.view(3, self.height, self.width)
        # TODO: If we need to shuffle the planes, might as well do the type conversion at the
        #       same time for better efficiency.
        rgb = brg[[1, 2, 0], ...]

        if self.dtype == torch.float32:
            return rgb.float().div_(255)
        elif self.dtype == torch.uint8:
            return (rgb * 255).round_().byte()
        raise NotImplementedError(f'Unsupported dtype: {self.dtype}')


class FffrBackendFactory(BackendFactory):
    def create(self, filename, device, dtype, backend_opts=None) -> FffrBackend:
        if backend_opts is None:
            backend_opts = {}
        return FffrBackend(filename, device, dtype, **backend_opts)
