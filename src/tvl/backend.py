from abc import ABC, abstractmethod


class Backend(ABC):
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


class BackendFactory(ABC):
    @abstractmethod
    def create(self, filename, device, dtype, backend_opts=None) -> Backend:
        pass
