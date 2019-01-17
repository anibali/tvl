import av
import torch

class PyAvBackendInstance:
    def __init__(self, filename, device):
        assert device.type == 'cpu'
        self.container = av.open(filename)
        self.generator = None

    def seek(self, time_secs):
        self.frame_reader.seek(time_secs)
        self.generator = None

    def read_frame_rgb(self):
        if self.generator is None:
            self.generator = self.container.decode(video=0)
        frame = next(self.generator)
        np_frame = frame.to_rgb().to_ndarray()
        return torch.from_numpy(np_frame).permute(2, 0, 1).float().div_(255)


class PyAvBackend:
    def create(self, filename, device):
        return PyAvBackendInstance(filename, torch.device(device))
