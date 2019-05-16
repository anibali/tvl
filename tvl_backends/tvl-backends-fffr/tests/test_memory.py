import torch
import pytest

from tvl_backends.fffr.memory import TorchMemManager


ALL_DEVICES = ['cpu']
# Add available GPU devices.
ALL_DEVICES.extend(f'cuda:{i}' for i in range(torch.cuda.device_count()))


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_allocate(device):
    mm = TorchMemManager(device)
    address = mm.allocate(7)
    assert len(mm.chunks) == 1
    assert mm[address].device == torch.device(device)
    assert len(mm[address]) == 7
    assert mm[address].data_ptr() == address


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_clear(device):
    mm = TorchMemManager(device)
    mm.allocate(7)
    assert len(mm.chunks) == 1
    mm.clear()
    assert len(mm.chunks) == 0


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_free(device):
    mm = TorchMemManager(device)
    addr1 = mm.allocate(4)
    addr2 = mm.allocate(4)
    assert len(mm.chunks) == 2
    mm.free(addr1)
    assert len(mm.chunks) == 1
    assert mm[addr2]


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_find_containing_chunk(device):
    mm = TorchMemManager(device)
    addresses = [mm.allocate(8) for _ in range(16)]
    address = addresses[5]
    assert mm.find_containing_chunk(address + 3) == mm[address]


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_tensor(device):
    mm = TorchMemManager(device)
    addresses = [mm.allocate(8) for _ in range(16)]
    address = addresses[5]
    tensor = mm.tensor(address + 3)
    assert tensor.shape == (5,)
