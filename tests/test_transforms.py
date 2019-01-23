import torch
import math
from numpy.testing import assert_allclose

from tvl.transforms import normalise, denormalise, resize


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


def test_resize(first_frame_image):
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
