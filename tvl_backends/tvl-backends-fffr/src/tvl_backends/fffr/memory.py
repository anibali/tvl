import pyfffr
import torch


def _dtype_bytes(dtype):
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    assert info.bits % 8 == 0
    return info.bits // 8


class TorchImageAllocator(pyfffr.ImageAllocator):
    def __init__(self, device, dtype):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.tensors = {}

    def allocate_frame(self, width, height, line_size):
        """Allocate memory for an image frame.

        The allocated memory shall have each plane aligned to 32-byte boundaries.

        Args:
            width (int): Width of the image in pixels.
            height (int): Height of the image in pixels.
            line_size (int): Size of a row of image pixels in bytes. The pixel data is expected
              to be left-aligned within this area.

        Returns:
            Address of the allocated memory.
        """
        align_bytes = 32  # Align memory to 32-byte boundaries.
        elem_size = _dtype_bytes(self.dtype)  # Size of a pixel channel element.
        assert align_bytes % elem_size == 0, 'alignment must be a multiple of element size'

        # Convert some sizes from bytes to element counts.
        line_elems = line_size // elem_size
        align_elems = align_bytes // elem_size

        # Allocate memory with extra space for planar alignment.
        n_padded_elems = 3 * (height * line_elems + align_elems)
        storage = torch.empty(n_padded_elems, device=self.device, dtype=self.dtype).storage()

        # Calculate memory offset and stride.
        ptr = storage.data_ptr()
        assert ptr % elem_size == 0
        aligned_ptr = (((ptr - 1) // align_bytes) * align_bytes + (align_bytes - ptr % align_bytes))
        storage_offset = (aligned_ptr - ptr) // elem_size
        plane_stride = ((height * line_elems - 1) // align_elems + 1) * align_elems

        # Create a tensor for viewing the allocated memory.
        tensor = torch.empty((0,), device=self.device, dtype=self.dtype)
        tensor.set_(storage, storage_offset=storage_offset, size=(3, height, width),
                    stride=(plane_stride, line_elems, 1))
        self.tensors[ptr] = tensor

        return ptr

    def free_frame(self, address):
        del self.tensors[address]

    def get_data_type(self):
        if self.dtype == torch.uint8:
            return pyfffr.ImageAllocator.UINT8
        if self.dtype == torch.float32:
            return pyfffr.ImageAllocator.FLOAT32
        raise Exception(f'unsupported dtype: {self.dtype}')

    def get_frame_tensor(self, address):
        return self.tensors[address]
