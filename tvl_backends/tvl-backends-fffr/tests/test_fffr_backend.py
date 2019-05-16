import PIL.Image
import numpy as np
import pytest


# def test_eof(backend):
#     backend.seek(2.0)
#     with pytest.raises(EOFError):
#         backend.read_frame()


def test_duration(backend):
    assert backend.duration == 2.0


def test_frame_rate(backend):
    assert backend.frame_rate == 25


def test_n_frames(backend):
    assert backend.n_frames == 50


def test_width(backend):
    assert backend.width == 1280


def test_height(backend):
    assert backend.height == 720
