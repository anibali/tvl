from abc import ABC, abstractmethod

import torch


class Backend(ABC):
    def __init__(self, filename, device, dtype, seek_threshold):
        """Create a video-reading backend instance for a particular video file.

        Args:
            filename: Path to the video file.
            device:
            dtype:
            seek_threshold (int): Hint for predicting when seeking to the next target frame
                would be faster than reading and discarding intermediate frames. Setting this value
                close to the video's GOP size should be a reasonable choice.
        """
        self.filename = filename
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.dtype = dtype
        self.seek_threshold = seek_threshold

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
        """The width of the image."""

    @property
    @abstractmethod
    def height(self):
        """The height of the image."""

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


class BackendFactory(ABC):
    @abstractmethod
    def create(self, filename, device, dtype, backend_opts=None) -> Backend:
        pass
