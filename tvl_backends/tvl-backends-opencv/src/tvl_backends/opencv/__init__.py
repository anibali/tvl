import cv2
import torch

from tvl.backend import Backend, BackendFactory


class OpenCvBackend(Backend):
    def __init__(self, filename, device):
        device = torch.device(device)
        assert device.type == 'cpu'
        self.cap = cv2.VideoCapture(filename)

    @property
    def duration(self):
        return self.n_frames / self.frame_rate

    @property
    def frame_rate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def n_frames(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def seek(self, time_secs):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, round(time_secs * self.frame_rate))

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(frame).permute(2, 0, 1).float().div_(255)
        else:
            raise EOFError()


class OpenCvBackendFactory(BackendFactory):
    def create(self, filename, device) -> OpenCvBackend:
        return OpenCvBackend(filename, device)
