import av
import torch

from tvl.backend import Backend, BackendFactory


class PyAvBackend(Backend):
    def __init__(self, filename, device):
        device = torch.device(device)
        assert device.type == 'cpu'
        self.container = av.open(filename)
        self.generator = None
        self.seek_time = None

    @property
    def duration(self):
        return self.container.duration / av.time_base

    @property
    def frame_rate(self):
        return self.container.streams.video[0].average_rate

    @property
    def n_frames(self):
        frames = self.container.streams.video[0].frames
        if frames > 0:
            return frames
        return int(self.duration * self.frame_rate)

    def seek(self, time_secs):
        self.container.seek(round(time_secs * av.time_base))
        self.seek_time = time_secs
        self.generator = None

    def read_frame(self):
        if self.generator is None:
            self.generator = self.container.decode(video=0)
        for frame in self.generator:
            if self.seek_time is None or frame.pts * frame.time_base >= self.seek_time:
                break
        else:
            raise EOFError()
        self.seek_time = None
        np_frame = frame.to_rgb().to_ndarray()
        return torch.from_numpy(np_frame).permute(2, 0, 1).float().div_(255)


class PyAvBackendFactory(BackendFactory):
    def create(self, filename, device) -> PyAvBackend:
        return PyAvBackend(filename, device)
