import pyfffr
import torch
from bisect import bisect_right, bisect_left


class _LazyDerivedList:
    def __init__(self, base_list: list, derive_value):
        self.base_list = base_list
        self.derive_value = derive_value

    def __getitem__(self, i):
        return self.derive_value(self.base_list[i])

    def __len__(self):
        return len(self.base_list)


class TorchMemManager(pyfffr.MemManager):
    """MemManager implementation which allocates Torch storage."""

    def __init__(self, device):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.chunks = []
        self.chunk_addresses = _LazyDerivedList(self.chunks, lambda e: e.data_ptr())

    def allocate(self, size):
        """Allocate a new chunk of memory.

        Args:
            size (int): The size (in bytes) of the memory chunk to allocate.

        Returns:
            int: The starting address of the allocated memory chunk.
        """
        storage = torch.empty(size, dtype=torch.uint8, device=self.device).storage()
        address = storage.data_ptr()
        index = bisect_right(self.chunk_addresses, address)
        self.chunks.insert(index, storage)
        return address

    def free(self, address):
        """Release reference to an allocated memory chunk."""
        del self.chunks[self._find_chunk_index(address)]

    def clear(self):
        """Release references to all allocated memory chunks."""
        self.chunks.clear()

    def _find_chunk_index(self, address):
        index = bisect_left(self.chunk_addresses, address)
        if index >= len(self.chunks) or self.chunk_addresses[index] != address:
            raise KeyError(f'no chunk starts at address 0x{address:08x}')
        return index

    def find_containing_chunk(self, address):
        index = bisect_right(self.chunk_addresses, address)
        if index > 0:
            chunk = self.chunks[index - 1]
            if 0 <= address - chunk.data_ptr() < len(chunk):
                return chunk
        raise KeyError(f'no chunk contains address 0x{address:08x}')

    def tensor(self, address, length=None):
        chunk = self.find_containing_chunk(address)
        offset = address - chunk.data_ptr()
        max_length = len(chunk) - offset
        if length is None:
            length = max_length
        if length > max_length:
            raise IndexError('specified length extends beyond chunk boundary')
        tensor = torch.empty((0,), dtype=torch.uint8, device=chunk.device)
        tensor.set_(chunk, offset, (length,))
        return tensor

    def __getitem__(self, address):
        return self.chunks[self._find_chunk_index(address)]
