import torch
import pytest

import pyfffr
from tvl_backends.fffr.memory import TorchImageAllocator


@pytest.mark.parametrize('dtype', ['float32', 'uint8'])
def test_allocate_frame(device, dtype):
    dtype = getattr(torch, dtype)
    mm = TorchImageAllocator(device, dtype)
    address = mm.allocate_frame(16, 9, 16, 32)
    assert address
    tensor = mm.get_frame_tensor(address)
    assert tensor.shape == (3, 9, 16)
    assert tensor.device == torch.device(device)
    assert tensor.dtype == dtype
    assert tensor.data_ptr() == address


@pytest.mark.parametrize('dtype,data_type_enum_value', [
    ('float32', pyfffr.ImageAllocator.FLOAT32),
    ('uint8', pyfffr.ImageAllocator.UINT8),
])
def test_get_data_type(device, dtype, data_type_enum_value):
    dtype = getattr(torch, dtype)
    allocator = TorchImageAllocator(device, dtype)
    assert allocator.get_data_type() == data_type_enum_value


def test_free_frame(device):
    mm = TorchImageAllocator(device, torch.uint8)
    addr1 = mm.allocate_frame(16, 9, 16, 32)
    addr2 = mm.allocate_frame(16, 9, 16, 32)
    assert len(mm.tensors) == 2
    mm.free_frame(addr1)
    assert len(mm.tensors) == 1
    assert mm.get_frame_tensor(addr2) is not None


def test_allocation_alignment(device):
    dtype = torch.float32
    allocator = TorchImageAllocator(device, dtype)
    addr = allocator.allocate_frame(1, 1, 4, 32)
    tensor = allocator.get_frame_tensor(addr)
    storage = tensor.storage()
    # Not using Storage#fill_ because of https://github.com/pytorch/pytorch/issues/22383
    for i in range(len(storage)):
        storage[i] = 0
    tensor.fill_(1)
    for i in range(3):
        index = tensor.storage_offset() + i * 8
        assert storage[index] == 1
    assert(len(storage) == 32)


def test_get_device_index():
    allocator = TorchImageAllocator('cuda', torch.float32)
    assert allocator.get_device_index() >= 0
    allocator = TorchImageAllocator('cpu', torch.float32)
    assert allocator.get_device_index() == -1
