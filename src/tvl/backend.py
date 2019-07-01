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

    def select_frames(self, frame_indices, seek_hint=3):
        # We will be loading unique frames in ascending index order.
        sorted_frame_indices = list(sorted(set(frame_indices)))

        pos = -(seek_hint + 1)
        seq_len = 0
        seq_keepers = []
        for frame_index in sorted_frame_indices:
            if frame_index - pos > seek_hint:
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
