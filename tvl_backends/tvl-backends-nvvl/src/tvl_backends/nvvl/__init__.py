import torch

import tvlnvvl
from tvl.backend import Backend, BackendFactory


class NvvlVideoLoader:
    """Wrapper for decoding video files on the GPU using NVVL.

    This code requires an NVIDIA GPU, and only decodes image data (no audio).

    Args:
        n_chans (int): number of colour channels in the source video images.
        height (int): height of the source video images.
        width (int): width of the source video images.
        device (torch.device): the CUDA device to use for decoding.
        dtype (torch.dtype): the data type of the tensor which will store the image data.
    """

    def __init__(self, n_chans, height, width, device, dtype=torch.uint8):
        assert device.type == 'cuda'
        self.device = device
        self.loader = tvlnvvl.nvvl_create_video_loader_with_log(self.device.index, tvlnvvl.LogLevel_Error)
        self.n_chans = n_chans
        self.height = height
        self.width = width
        assert dtype in {torch.uint8, torch.float16, torch.float32}, \
            'dtype must be uint8, float16, or float32'
        self.dtype = dtype

    def _create_pic_layer(self, n_frames, crop_bounds, scale):
        n, c, h, w = n_frames, self.n_chans, self.height, self.width
        pic = tvlnvvl.NVVL_PicLayer()
        pic.desc.count = n
        pic.desc.channels = c
        if scale:
            pic.desc.scale_width = int(round(w * scale))
            pic.desc.scale_height = int(round(h * scale))
        if crop_bounds:
            pic.desc.crop_x = int(round(crop_bounds[0]))
            pic.desc.crop_y = int(round(crop_bounds[1]))
            pic.desc.width = int(round(crop_bounds[2]))
            pic.desc.height = int(round(crop_bounds[3]))
        else:
            pic.desc.width = pic.desc.scale_width if scale else w
            pic.desc.height = pic.desc.scale_height if scale else h
        return pic

    def _add_tensor_to_pic_layer(self, pic, tensor):
        pic.desc.stride.n = tensor.stride(1)
        pic.desc.stride.c = tensor.stride(0)
        pic.desc.stride.y = tensor.stride(2)
        pic.desc.stride.x = tensor.stride(3)
        if tensor.dtype == torch.uint8:
            pic.type = tvlnvvl.PDT_BYTE
        elif tensor.dtype == torch.float16:
            pic.type = tvlnvvl.PDT_HALF
        elif tensor.dtype == torch.float32:
            pic.type = tvlnvvl.PDT_FLOAT
        else:
            pic.type = tvlnvvl.PDT_NONE
        pic.desc.scale_method = tvlnvvl.ScaleMethod_Nearest
        pic.data = tensor.data_ptr()
        return pic

    def _load_sequence_into_tensor(self, tensor, filename, start_frame, pic):
        sequence = tvlnvvl.nvvl_create_sequence_device(pic.desc.count, self.device.index)
        self._add_tensor_to_pic_layer(pic, tensor)
        layer_name = 'pixels'  # Name for debugging purposes
        tvlnvvl.nvvl_set_layer(sequence, pic, layer_name)
        tvlnvvl.nvvl_read_sequence(self.loader, filename, start_frame, pic.desc.count)
        tvlnvvl.nvvl_receive_frames_sync(self.loader, sequence)
        return tensor

    def load_sequence(self, filename, start_frame, n_frames, crop_bounds=None, scale=None, tensor=None):
        """Load a continuous sequence of frames from a video file.

        Args:
            filename (str): path to video file.
            start_frame (int): index of the starting frame (frame 0 is the first frame).
            n_frames (int): number of frames to read.
            crop_bounds (list of float): bounding box (x, y, width, height) of area to crop.
            scale (float): scale factor to resize the frames by.
            tensor (torch.Tensor): tensor to store the decoded frames in.

        Returns:
            torch.Tensor: the decoded sequence of frames.
        """
        pic = self._create_pic_layer(n_frames, crop_bounds, scale)
        if tensor is None:
            tensor = torch.empty(
                pic.desc.channels, pic.desc.count, pic.desc.height, pic.desc.width,
                dtype=self.dtype, device=self.device)
        return self._load_sequence_into_tensor(tensor, filename, start_frame, pic)

    def close(self):
        tvlnvvl.nvvl_destroy_video_loader(self.loader)
        self.loader = None


class NvvlBackend(Backend):
    def __init__(self, filename, device, dtype, scale=None):
        device = torch.device(device)
        assert device.type == 'cuda'

        self.info = tvlnvvl.VideoInfo(filename)

        self.filename = filename
        self.device = device
        self.loader = NvvlVideoLoader(3, self.info.get_height(), self.info.get_width(),
                                      self.device, dtype=dtype)
        self.dtype = dtype

        self.cur_frame_index = 0
        self.scale = scale

    @property
    def duration(self):
        return self.info.get_duration()

    @property
    def frame_rate(self):
        return self.info.get_frame_rate()

    @property
    def n_frames(self):
        return self.info.get_number_of_frames()

    @property
    def width(self):
        return self.info.get_width()

    @property
    def height(self):
        return self.info.get_height()

    def seek(self, time_secs):
        self.seek_to_frame(round(time_secs * self.frame_rate))

    def seek_to_frame(self, frame_index):
        self.cur_frame_index = frame_index

    def read_frame(self):
        # NOTE: This is a slow way to read sequential frames. It's a much better idea to read
        #       frames in batches using read_frames instead.
        return self.read_frames(1)[0]

    def read_frames(self, n):
        if n == 0:
            return []
        if self.cur_frame_index + n > self.n_frames:
            raise EOFError()
        tensor = self.loader.load_sequence(self.filename, self.cur_frame_index, n, scale=self.scale)
        if self.dtype.is_floating_point:
            tensor.div_(255)
        frames = list(tensor.unbind(1))
        self.cur_frame_index += n
        return frames


class NvvlBackendFactory(BackendFactory):
    def create(self, filename, device, dtype, backend_opts=None) -> NvvlBackend:
        # TODO: It should be possible to share NVVL video loaders across video files
        #       of the same resolution.
        if backend_opts is None:
            backend_opts = {}
        return NvvlBackend(filename, device, dtype, **backend_opts)
