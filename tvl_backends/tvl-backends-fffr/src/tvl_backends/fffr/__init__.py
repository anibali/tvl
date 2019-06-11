import torch

import pyfffr
from tvl.backend import Backend, BackendFactory
from tvl_backends.fffr.memory import TorchMemManager


class FffrBackend(Backend):
    def __init__(self, filename, device, dtype):
        device = torch.device(device)
        if device.type == 'cuda':
            device_index = device.index
        else:
            device_index = -1

        mem_manager = TorchMemManager(device)
        frame_reader = pyfffr.TvFFFrameReader(mem_manager, filename, device_index)
        # We need to hold a reference to mem_manager for at least as long as the TvFFFrameReader
        # that uses it is around, since we retain ownership of the TorchMemManager object.
        setattr(frame_reader, '__mem_manager_ref', mem_manager)

        self.mem_manager = mem_manager
        self.frame_reader = frame_reader
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

    @property
    def frame_size(self):
        return self.frame_reader.get_frame_size()

    def seek(self, time_secs):
        self.frame_reader.seek(time_secs)

    def read_frame(self):
        ptr = self.frame_reader.read_frame()
        if not ptr:
            raise EOFError()

        brg = self.mem_manager.tensor(int(ptr), self.frame_reader.get_frame_size())
        self.mem_manager.free(int(ptr))  # Release reference held by the memory manager.
        brg = brg.view(3, self.height, self.width)

        rgb = torch.empty(brg.shape, dtype=self.dtype, device=self.mem_manager.device)
        rgb[0] = brg[1]
        rgb[1] = brg[2]
        rgb[2] = brg[0]

        if self.dtype == torch.float32:
            return rgb.div_(255)
        elif self.dtype == torch.uint8:
            return rgb
        raise NotImplementedError(f'Unsupported dtype: {self.dtype}')


class FffrBackendFactory(BackendFactory):
    def create(self, filename, device, dtype, backend_opts=None) -> FffrBackend:
        if backend_opts is None:
            backend_opts = {}
        return FffrBackend(filename, device, dtype, **backend_opts)
