from abc import ABC, abstractmethod


class BackendInstance(ABC):
    @property
    @abstractmethod
    def duration(self):
        """The duration of the video (in seconds)."""

    @property
    @abstractmethod
    def frame_rate(self):
        """The frame rate of the video (in frames per second)."""

    @property
    def n_frames(self):
        """The number of frames in the video."""
        return int(self.duration * self.frame_rate)

    @abstractmethod
    def seek(self, time_secs):
        """Seek to the specified time in the video file."""

    @abstractmethod
    def read_frame(self):
        """Read a single video frame as an RGB PyTorch tensor."""


class Backend(ABC):
    @abstractmethod
    def create(self, filename, device) -> BackendInstance:
        pass
