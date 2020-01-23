import os.path
from abc import ABC, abstractmethod

import torch

from tvl.transforms import resize


class Backend(ABC):
    def __init__(self, filename, device, dtype, seek_threshold, out_width, out_height):
        """Create a video-reading backend instance for a particular video file.

        Args:
            filename: Path to the video file.
            device:
            dtype:
            seek_threshold (int): Hint for predicting when seeking to the next target frame
                would be faster than reading and discarding intermediate frames. Setting this value
                close to the video's GOP size should be a reasonable choice.
            out_width (int): Desired output width of read frames.
            out_height (int): Desired output height of read frames.
        """
        self.filename = filename
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.dtype = dtype
        self.seek_threshold = seek_threshold
        self._out_width = out_width
        self._out_height = out_height

    @property
    @abstractmethod
    def duration(self):
        """The duration of the video (in seconds)."""

    @property
    @abstractmethod
    def frame_rate(self):
        """The frame rate of the video (in frames per second)."""

    @property
    @abstractmethod
    def n_frames(self):
        """The number of frames in the video."""

    @property
    @abstractmethod
    def width(self):
        """The original width of a frame in the video file."""

    @property
    @abstractmethod
    def height(self):
        """The original height of a frame in the video file."""

    @property
    def out_width(self):
        """The width of the output image after reading."""
        if self._out_width > 0:
            return self._out_width
        else:
            return self.width

    @property
    def out_height(self):
        """The height of the output image after reading."""
        if self._out_height > 0:
            return self._out_height
        else:
            return self.height

    @abstractmethod
    def seek(self, time_secs):
        """Seek to the specified time in the video file."""

    def seek_to_frame(self, frame_index):
        self.seek(frame_index / self.frame_rate)

    @abstractmethod
    def read_frame(self):
        """Read a single video frame as an RGB PyTorch tensor."""

    def read_frames(self, n):
        return [self.read_frame() for _ in range(n)]

    def select_frames(self, frame_indices):
        # We will be loading unique frames in ascending index order.
        sorted_frame_indices = list(sorted(set(frame_indices)))

        pos = -(self.seek_threshold + 1)
        seq_len = 0
        seq_keepers = []
        for frame_index in sorted_frame_indices:
            if frame_index - pos > self.seek_threshold:
                # Read previous sequence
                if seq_len > 0:
                    frames = self.read_frames(seq_len)
                    for i in seq_keepers:
                        yield frames[i]
                    seq_keepers.clear()
                    seq_len = 0
                # Skip to desired location by seeking.
                self.seek_to_frame(frame_index)
            else:
                # Skip to desired location by reading and discarding intermediate frames.
                while pos < frame_index:
                    seq_len += 1
                    pos += 1
            # Read the frame that we care about.
            seq_keepers.append(seq_len)
            seq_len += 1
            pos = frame_index + 1
        if seq_len > 0:
            frames = self.read_frames(seq_len)
            for i in seq_keepers:
                yield frames[i]

    def select_frame(self, frame_index):
        return next(self.select_frames([frame_index]))

    def _postprocess_frame(self, rgb: torch.Tensor):
        """Postprocess an RGB image tensor to have the expected dtype and size."""
        if self.dtype == torch.float32:
            if not rgb.is_floating_point():
                rgb = rgb.to(self.dtype).div_(255)
            else:
                rgb = rgb.to(self.dtype)
        elif self.dtype == torch.uint8:
            if rgb.is_floating_point():
                rgb = rgb.mul_(255).to(self.dtype)
            else:
                rgb = rgb.to(self.dtype)
        else:
            raise NotImplementedError(f'Unsupported dtype: {self.dtype}')
        if self._out_height > 0 or self._out_width > 0:
            rgb = resize(rgb, (self.out_height, self.out_width))
        return rgb


class BackendFactory(ABC):
    @abstractmethod
    def create(self, filename, device, dtype, backend_opts=None) -> Backend:
        pass
