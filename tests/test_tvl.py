import tvl


def test_vl_read_frame(dummy_backend):
    vl = tvl.VideoLoader('', dummy_backend.device)
    assert vl.read_frame() == dummy_backend.frames[0]


def test_vl_read_frames(dummy_backend):
    vl = tvl.VideoLoader('', dummy_backend.device)
    assert list(vl.read_frames()) == dummy_backend.frames
