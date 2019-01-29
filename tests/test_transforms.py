import math

import hypothesis
import numpy as np
import torch
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import booleans, one_of, just
from torch.testing import assert_allclose

from tvl.transforms import normalise, denormalise, resize, crop, flip, rotate

DENORMALISED_IMAGE = torch.tensor([math.sqrt(3), -math.sqrt(3)]).add_(5).repeat(3, 2, 1)
MEAN = [5.0, 5.0, 5.0]
STDDEV = [2.0, 2.0, 2.0]
NORMALISED_IMAGE = torch.tensor([math.sqrt(3) / 2, -math.sqrt(3) / 2]).repeat(3, 2, 1)


def test_normalise():
    denormalised = DENORMALISED_IMAGE.clone()
    normalised = normalise(denormalised, MEAN, STDDEV, inplace=False)
    assert_allclose(denormalised, DENORMALISED_IMAGE)  # inplace=False should be respected
    assert_allclose(normalised, NORMALISED_IMAGE)  # Result should be correct


def test_normalise_inplace():
    denormalised = DENORMALISED_IMAGE.clone()
    normalised = normalise(denormalised, MEAN, STDDEV, inplace=True)
    assert normalised.data_ptr() == denormalised.data_ptr()  # inplace=True should be respected
    assert_allclose(normalised, NORMALISED_IMAGE)  # Result should be correct


def test_denormalise():
    normalised = NORMALISED_IMAGE.clone()
    denormalised = denormalise(normalised, MEAN, STDDEV, inplace=False)
    assert_allclose(normalised, NORMALISED_IMAGE)  # inplace=False should be respected
    assert_allclose(denormalised, DENORMALISED_IMAGE)  # Result should be correct


def test_denormalise_inplace():
    normalised = NORMALISED_IMAGE.clone()
    denormalised = denormalise(normalised, MEAN, STDDEV, inplace=True)
    assert denormalised.data_ptr() == normalised.data_ptr()  # inplace=True should be respected
    assert_allclose(denormalised, DENORMALISED_IMAGE)  # Result should be correct


def test_resize():
    inp = torch.FloatTensor([[
        [1, 0],
        [0, 1],
    ]])
    expected = torch.FloatTensor([[
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ]])
    actual = resize(inp, (4, 4), mode='nearest')
    assert_allclose(actual, expected)


def test_crop():
    inp = torch.FloatTensor([[
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ]])
    expected = torch.FloatTensor([[
        [1, 0],
        [0, 1],
        [0, 1],
    ]])
    actual = crop(inp, 1, 1, 3, 2)
    assert_allclose(actual, expected)


def test_crop_padded():
    inp = torch.FloatTensor([[
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ]])
    expected = torch.FloatTensor([[
        [2, 2, 2, 2, 2],
        [1, 0, 0, 2, 2],
    ]])
    actual = crop(inp, -1, 1, 2, 5, padding_mode='constant', fill=2)
    assert_allclose(actual, expected)


@hypothesis.given(
    data=arrays(np.float32, array_shapes(min_dims=2, max_dims=4)),
    horizontal=booleans(),
    vertical=booleans(),
    device=one_of(just(torch.device(e)) for e in ['cpu', 'cuda:0'])
)
def test_flip_involution(data, horizontal, vertical, device):
    inp = torch.from_numpy(data).to(device)
    flipped_once = flip(inp, horizontal, vertical)
    flipped_twice = flip(flipped_once, horizontal, vertical)
    assert_allclose(flipped_twice, inp)


def test_flip_horizontal():
    inp = torch.FloatTensor([[
        [1, 2],
        [3, 4],
    ]])
    expected = torch.FloatTensor([[
        [2, 1],
        [4, 3],
    ]])
    actual = flip(inp, horizontal=True)
    assert_allclose(actual, expected)


def test_flip_vertical():
    inp = torch.FloatTensor([[
        [1, 2],
        [3, 4],
    ]])
    expected = torch.FloatTensor([[
        [3, 4],
        [1, 2],
    ]])
    actual = flip(inp, vertical=True)
    assert_allclose(actual, expected)


def test_rotate():
    inp = torch.FloatTensor([[
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ]])
    expected = torch.FloatTensor([[
        [0, 0],
        [4, 6],
        [3, 5],
        [0, 0],
    ]])
    actual = rotate(inp, 90)
    assert_allclose(actual, expected)


def test_rotate_cuda():
    inp = torch.cuda.FloatTensor([[
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ]])
    expected = torch.cuda.FloatTensor([[
        [0, 0],
        [4, 6],
        [3, 5],
        [0, 0],
    ]])
    actual = rotate(inp, 90)
    assert_allclose(actual, expected)
