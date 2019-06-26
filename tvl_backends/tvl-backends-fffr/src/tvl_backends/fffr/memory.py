import pyfffr
import torch


def _dtype_bytes(dtype):
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    assert info.bits % 8 == 0
    return info.bits // 8

def _align_to(value, alignment):
    return (value + (alignment - 1)) & ~(alignment - 1)


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
        """
        align_bytes = 32
        elem_size = _dtype_bytes(self.dtype)
        assert align_bytes % elem_size == 0
        line_elems = line_size // elem_size
        align_elems = align_bytes // elem_size

        n_padded_elems = 3 * (height * line_elems + align_elems)
        storage = torch.empty(n_padded_elems, device=self.device, dtype=self.dtype).storage()
        ptr = storage.data_ptr()
        assert ptr % elem_size == 0
        aligned_ptr = _align_to(ptr, align_bytes)
        storage_offset = (aligned_ptr - ptr) // elem_size
        plane_stride_bytes = _align_to(height * line_size, align_bytes)
        plane_stride = plane_stride_bytes // elem_size

        tensor = torch.empty((0,), device=self.device, dtype=self.dtype)
        tensor.set_(storage, storage_offset=storage_offset, size=(3, height, width),
                    stride=(plane_stride, line_elems, 1))
        address = tensor.data_ptr()

        self.tensors[address] = tensor
        return address

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
