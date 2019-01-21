from abc import ABC, abstractmethod


class BackendInstance(ABC):
    @abstractmethod
    def seek(self, time_secs):
        pass

    @abstractmethod
    def read_frame_rgb(self):
        pass


class Backend(ABC):
    @abstractmethod
    def create(self, filename, device) -> BackendInstance:
        pass
