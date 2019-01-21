import av
import torch

from .common import BackendInstance, Backend


class PyAvBackendInstance(BackendInstance):
    def __init__(self, filename, device):
        assert device.type == 'cpu'
        self.container = av.open(filename)
        self.generator = None
        self.seek_time = None

    def seek(self, time_secs):
        self.container.seek(round(time_secs * av.time_base))
        self.seek_time = time_secs
        self.generator = None

    def read_frame_rgb(self):
        if self.generator is None:
            self.generator = self.container.decode(video=0)
        for frame in self.generator:
            if self.seek_time is None or frame.pts * frame.time_base >= self.seek_time:
                break
        else:
            raise EOFError('no more frames.')
        self.seek_time = None
        np_frame = frame.to_rgb().to_ndarray()
        return torch.from_numpy(np_frame).permute(2, 0, 1).float().div_(255)


class PyAvBackend(Backend):
    def create(self, filename, device):
        return PyAvBackendInstance(filename, torch.device(device))
