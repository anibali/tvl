import cv2
import numpy as np
import torch

from tvl.backend import Backend, BackendFactory


class OpenCvBackend(Backend):
    def __init__(self, filename, device, dtype, *, seek_threshold=3, out_width=0, out_height=0):
        super().__init__(filename, device, dtype, seek_threshold, out_width, out_height)
        assert self.device.type == 'cpu'
        self.cap = cv2.VideoCapture(self.filename)

    @property
    def duration(self):
        return self.n_frames / self.frame_rate

    @property
    def frame_rate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def n_frames(self):
        return int(round(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    @property
    def width(self):
        return int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    @property
    def height(self):
        return int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def seek_to_frame(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def seek(self, time_secs):
        self.seek_to_frame(int(round(time_secs * self.frame_rate)))

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_bytes = torch.from_numpy(np.moveaxis(frame, -1, 0))
            return self._postprocess_frame(rgb_bytes)
        else:
            raise EOFError()


class OpenCvBackendFactory(BackendFactory):
    def create(self, filename, device, dtype, backend_opts=None) -> OpenCvBackend:
        if backend_opts is None:
            backend_opts = {}
        return OpenCvBackend(filename, device, dtype, **backend_opts)
