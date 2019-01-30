from unittest.mock import call

import tvl


def test_vl_read_frame(dummy_backend):
    vl = tvl.VideoLoader('', dummy_backend.device)
    assert vl.read_frame() == dummy_backend.frames[0]


def test_vl_read_frames(dummy_backend):
    vl = tvl.VideoLoader('', dummy_backend.device)
    assert list(vl.read_frames()) == dummy_backend.frames


def test_vl_pick_frames_sequental(dummy_backend, mocker):
    vl = tvl.VideoLoader('', dummy_backend.device)
    mocked_seek = mocker.patch.object(vl, 'seek')
    vl.pick_frames([24, 26, 25])
    # When frame indices are dense, only one seek should occur
    mocked_seek.assert_called_once_with(2.4)


def test_vl_pick_frames_random(dummy_backend, mocker):
    vl = tvl.VideoLoader('', dummy_backend.device)
    mocked_seek = mocker.patch.object(vl, 'seek')
    vl.pick_frames([5, 45, 25])
    # When frame indices are sparse, multiple seeks should occur
    assert mocked_seek.mock_calls == [call(0.5), call(2.5), call(4.5)]


def test_vl_pick_frames_mixed(dummy_backend, mocker):
    vl = tvl.VideoLoader('', dummy_backend.device)
    mocked_seek = mocker.patch.object(vl, 'seek')
    vl.pick_frames([1, 2, 10, 12])
    assert mocked_seek.mock_calls == [call(0.1), call(1.0)]
