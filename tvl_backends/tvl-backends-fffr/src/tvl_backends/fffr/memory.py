import warnings

import pyfffr
import torch


def _dtype_bytes(dtype):
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    assert info.bits % 8 == 0
    return info.bits // 8


def _align(value, alignment):
    """Find the smallest multiple of `alignment` that is at least as large as `value`."""
    return ((value - 1) // alignment + 1) * alignment


class TorchImageAllocator(pyfffr.ImageAllocator):
    def __init__(self, device, dtype):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.tensors = {}

    def allocate_frame(self, width, height, line_size, alignment):
        """Allocate memory for an image frame.

        The allocated memory shall have each line aligned according the the specified alignment.

        Args:
            width (int): Width of the image in pixels.
            height (int): Height of the image in pixels.
            line_size (int): Size of a row of image pixels in bytes. The pixel data is expected
              to be left-aligned within this area.
            alignment (int): Memory alignment in bytes.

        Returns:
            Address of the allocated memory.
        """
        elem_size = _dtype_bytes(self.dtype)  # Size of a pixel channel element.
        assert alignment % elem_size == 0, 'alignment must be a multiple of element size'

        # Align the line size.
        line_size = _align(line_size, alignment)

        # Convert some sizes from bytes to element counts.
        line_elems = line_size // elem_size
        align_elems = alignment // elem_size

        # Allocate memory with extra space for starting pointer alignment.
        n_padded_elems = 3 * (height * line_elems) + align_elems
        storage = torch.empty(n_padded_elems, device=self.device, dtype=self.dtype).storage()

        # Calculate memory offset and stride.
        ptr = storage.data_ptr()
        aligned_ptr = _align(ptr, alignment)
        assert (aligned_ptr - ptr) % elem_size == 0
        storage_offset = (aligned_ptr - ptr) // elem_size
        plane_stride = height * line_elems

        # Create a tensor for viewing the allocated memory.
        tensor = torch.empty((0,), device=self.device, dtype=self.dtype)
        tensor.set_(storage, storage_offset=storage_offset, size=(3, height, width),
                    stride=(plane_stride, line_elems, 1))
        self.tensors[ptr] = tensor

        return ptr

    def free_frame(self, address):
        try:
            del self.tensors[int(address)]
        except KeyError:
            warnings.warn('Skipped an attempt to free unrecognised memory.')

    def get_data_type(self):
        if self.dtype == torch.uint8:
            return pyfffr.ImageAllocator.UINT8
        if self.dtype == torch.float32:
            return pyfffr.ImageAllocator.FLOAT32
        raise Exception(f'unsupported dtype: {self.dtype}')

    def get_device_index(self):
        if self.device.type == 'cuda':
            if self.device.index is None:
                return torch.cuda.current_device()
            else:
                return self.device.index
        return -1

    def get_frame_tensor(self, address):
        return self.tensors[address]
