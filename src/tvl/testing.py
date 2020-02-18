import PIL.Image
import numpy as np
import torch
from numpy.testing import assert_allclose


def assert_same_image(actual, expected, atol=50, allow_mismatch=0.0):
    if torch.is_tensor(actual):
        if actual.is_floating_point():
            actual = actual * 255
        actual = actual.to(device='cpu', dtype=torch.uint8)
        actual = PIL.Image.fromarray(actual.permute(1, 2, 0).numpy(), 'RGB')
    if allow_mismatch > 0:
        close_elements = np.isclose(actual, expected, rtol=0, atol=atol)
        if np.sum(close_elements) / close_elements.size > (1.0 - allow_mismatch):
            return
    assert_allclose(actual, expected, rtol=0, atol=atol)
