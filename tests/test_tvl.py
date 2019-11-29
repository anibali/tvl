from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from random import uniform
from time import sleep
from unittest.mock import call

import pytest
import torch

import tvl
from tvl.backend import Backend


def test_vl_read_frame(dummy_backend_factory_cpu):
    vl = tvl.VideoLoader('', dummy_backend_factory_cpu.device)
    assert vl.read_frame() == dummy_backend_factory_cpu.frames[0]


def test_vl_read_all_frames(dummy_backend_factory_cpu):
    vl = tvl.VideoLoader('', dummy_backend_factory_cpu.device)
    assert list(vl.read_all_frames()) == dummy_backend_factory_cpu.frames


def test_vl_remaining_frames(dummy_backend_factory_cpu):
    vl = tvl.VideoLoader('', dummy_backend_factory_cpu.device)
    vl.read_frame()
    assert list(vl.remaining_frames()) == dummy_backend_factory_cpu.frames[1:]


def test_vl_select_frames_sequental(dummy_backend_factory_cpu, mocker):
    vl = tvl.VideoLoader('', dummy_backend_factory_cpu.device)
    mocked_seek = mocker.patch.object(vl.backend, 'seek_to_frame')
    list(vl.select_frames([24, 26, 25]))
    # When frame indices are dense, only one seek should occur
    mocked_seek.assert_called_once_with(24)


def test_vl_select_frames_random(dummy_backend_factory_cpu, mocker):
    vl = tvl.VideoLoader('', dummy_backend_factory_cpu.device)
    mocked_seek = mocker.patch.object(vl.backend, 'seek_to_frame')
    list(vl.select_frames([5, 45, 25]))
    # When frame indices are sparse, multiple seeks should occur
    assert mocked_seek.mock_calls == [call(5), call(25), call(45)]


def test_vl_select_frames_mixed(dummy_backend_factory_cpu, mocker):
    vl = tvl.VideoLoader('', dummy_backend_factory_cpu.device)
    mocked_seek = mocker.patch.object(vl.backend, 'seek_to_frame')
    list(vl.select_frames([1, 2, 10, 12]))
    assert mocked_seek.mock_calls == [call(1), call(10)]


@pytest.mark.parametrize('device,expected', [
    ('cuda:0', 'cuda:0'),
    ('cuda:1', 'cuda:1'),
    ('cuda', 'cuda:0'),
])
def test_backend_device(video_filename, device, expected, mocker):
    mocker.patch.object(Backend, '__abstractmethods__', new_callable=set)
    backend = Backend(video_filename, device, torch.float32, 3, 0, 0)
    assert str(backend.device) == expected


def test_video_loader_pool(video_filename):
    vlp = tvl.VideoLoaderPool({
        'cuda:0': 2,
        'cpu': 10,
    })
    expected = ['cuda:0', 'cuda:0', 'cpu', 'cpu', 'cpu']

    with vlp.loader(video_filename) as vl:
        assert str(vl.backend.device) == expected[0]

    with ExitStack() as stack:
        vls = [stack.enter_context(vlp.loader(video_filename)) for _ in expected]
        assert [str(vl.backend.device) for vl in vls] == expected


def test_video_loader_pool_threaded(video_filename):
    vlp = tvl.VideoLoaderPool({
        'cuda:0': 1,
        'cpu': 2,
    })

    def get():
        with vlp.loader(video_filename) as vl:
            sleep(0.05 + uniform(0, 0.01))
            return str(vl.backend.device)

    for _ in range(10):
        executor = ThreadPoolExecutor(max_workers=9)
        jobs = [executor.submit(get) for _ in range(9)]
        actual = [job.result() for job in jobs]
        assert actual.count('cuda:0') == 3
        assert actual.count('cpu') == 6
