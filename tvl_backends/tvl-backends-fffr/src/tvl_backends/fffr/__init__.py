import torch

import pyfffr
from tvl.backend import Backend, BackendFactory
from tvl_backends.fffr.memory import TorchImageAllocator


class FffrBackend(Backend):
    def __init__(self, filename, device, dtype):
        device = torch.device(device)
        if device.type == 'cuda':
            device_index = device.index
        else:
            device_index = -1

        allocator_dtype = dtype
        # The FFFR backend does not currently support direct conversion to float32 for software
        # decoding, so we will read as uint8 and do the data type conversion afterwards.
        if device.type == 'cpu' and dtype != torch.uint8:
            allocator_dtype = torch.uint8

        image_allocator = TorchImageAllocator(device, allocator_dtype)
        frame_reader = pyfffr.TvFFFrameReader(image_allocator, filename, device_index)
        # We need to hold a reference to image_allocator for at least as long as the
        # TvFFFrameReader that uses it is around, since we retain ownership of image_allocator.
        setattr(frame_reader, '__image_allocator_ref', image_allocator)

        self.image_allocator = image_allocator
        self.frame_reader = frame_reader
        self.dtype = dtype
        self._at_eof = False

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
        try:
            self.frame_reader.seek(time_secs)
            self._at_eof = False
        except RuntimeError:
            if time_secs < self.duration - (1.0 / self.frame_rate + 1e-9):
                raise
            self._at_eof = True

    def read_frame(self):
        if self._at_eof:
            raise EOFError()

        ptr = self.frame_reader.read_frame()
        if not ptr:
            raise EOFError()

        rgb_tensor = self.image_allocator.get_frame_tensor(int(ptr))
        self.image_allocator.free_frame(int(ptr))  # Release reference held by the memory manager.

        if self.dtype == torch.float32:
            if self.image_allocator.dtype != torch.float32:
                return rgb_tensor.to(self.dtype).div_(255)
            return rgb_tensor
        elif self.dtype == torch.uint8:
            return rgb_tensor
        raise NotImplementedError(f'Unsupported dtype: {self.dtype}')


class FffrBackendFactory(BackendFactory):
    def create(self, filename, device, dtype, backend_opts=None) -> FffrBackend:
        if backend_opts is None:
            backend_opts = {}
        return FffrBackend(filename, device, dtype, **backend_opts)
