import cv2
import torch

from tvl.backend import Backend, BackendFactory


class OpenCvBackend(Backend):
    def __init__(self, filename, device, dtype):
        device = torch.device(device)
        assert device.type == 'cpu'
        self.cap = cv2.VideoCapture(filename)
        self.dtype = dtype

    @property
    def duration(self):
        return self.n_frames / self.frame_rate

    @property
    def frame_rate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def n_frames(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def width(self):
        return int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    @property
    def height(self):
        return int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def seek(self, time_secs):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, round(time_secs * self.frame_rate))

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255)
            if self.dtype == torch.float32:
                return rgb
            elif self.dtype == torch.uint8:
                return (rgb * 255).round_().byte()
            raise NotImplementedError(f'Unsupported dtype: {self.dtype}')
        else:
            raise EOFError()


class OpenCvBackendFactory(BackendFactory):
    def create(self, filename, device, dtype, backend_opts=None) -> OpenCvBackend:
        return OpenCvBackend(filename, device, dtype)
