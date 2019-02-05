"""Functions for transforming image data stored in PyTorch tensors.

This module is necessary since most of the transformations provided by the `torchvision` package
are applicable for PIL.Image images only. Since tvl may load video frames on the GPU, we want
to be able to take the computation to the data rather than moving the images to and from main
memory.

As an additional benefit, these functions are defined such that they also work in batched mode,
which is especially useful for videos.
"""

import math
from typing import Sequence

import torch
from torch.nn.functional import interpolate
from torchgeometry.imgwarp import warp_affine


def normalise(tensor, mean, stddev, inplace=False):
    """Normalise the image with channel-wise mean and standard deviation.

    Args:
        tensor (Tensor): The image tensor to be normalised.
        mean (Sequence of float): Means for each channel.
        stddev (Sequence of float): Standard deviations for each channel.
        inplace (bool): Perform normalisation in-place.

    Returns:
        Tensor: The normalised image tensor.
    """
    mean = torch.as_tensor(mean, device=tensor.device)[..., :, None, None]
    stddev = torch.as_tensor(stddev, device=tensor.device)[..., :, None, None]

    if inplace:
        tensor.sub_(mean)
    else:
        tensor = tensor.sub(mean)

    tensor.div_(stddev)
    return tensor


def denormalise(tensor, mean, stddev, inplace=False):
    """Denormalise the image with channel-wise mean and standard deviation.

    Args:
        tensor (Tensor): The image tensor to be denormalised.
        mean (Sequence of float): Means for each channel.
        stddev (Sequence of float): Standard deviations for each channel.
        inplace (bool): Perform denormalisation in-place.

    Returns:
        Tensor: The denormalised image tensor.
    """
    mean = torch.as_tensor(mean, device=tensor.device)[..., :, None, None]
    stddev = torch.as_tensor(stddev, device=tensor.device)[..., :, None, None]

    if inplace:
        return tensor.mul_(stddev).add_(mean)
    else:
        return torch.addcmul(mean, tensor, stddev)


def resize(tensor, size, mode='bilinear'):
    """Resize the image.

    Args:
        tensor (Tensor): The image tensor to be resized.
        size (tuple of int): Size of the resized image (height, width).
        mode (str): The pixel sampling interpolation mode to be used.

    Returns:
        Tensor: The resized image tensor.
    """
    is_unbatched = tensor.ndimension() == 3
    if is_unbatched:
        tensor = tensor.unsqueeze(0)
    align_corners = None
    if mode in {'linear', 'bilinear', 'trilinear'}:
        align_corners = False
    resized = interpolate(tensor, size=size, mode=mode, align_corners=align_corners)
    if is_unbatched:
        resized = resized.squeeze(0)
    return resized


def crop(tensor, t, l, h, w, padding_mode='constant', fill=0):
    """Crop the image, padding out-of-bounds regions.

    Args:
        tensor (Tensor): The image tensor to be cropped.
        t (int): Top pixel coordinate.
        l (int): Left pixel coordinate.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
        padding_mode (str): Padding mode (currently "constant" is the only valid option).
        fill (float): Fill value to use with constant padding.

    Returns:
        Tensor: The cropped image tensor.
    """
    # If the crop region is wholly within the image, simply narrow the tensor.
    if t >= 0 and l >= 0 and h <= tensor.size(-2) and w <= tensor.size(-1):
        return tensor[..., t:t+h, l:l+w]

    if padding_mode == 'constant':
        result = torch.full((*tensor.size()[:-2], h, w), fill)
    else:
        raise Exception('crop only supports "constant" padding currently.')

    sx1 = l
    sy1 = t
    sx2 = l + w
    sy2 = t + h
    dx1 = 0
    dy1 = 0

    if sx1 < 0:
        dx1 = -sx1
        w += sx1
        sx1 = 0

    if sy1 < 0:
        dy1 = -sy1
        h += sy1
        sy1 = 0

    if sx2 >= tensor.size(-1):
        w -= sx2 - tensor.size(-1)

    if sy2 >= tensor.size(-2):
        h -= sy2 - tensor.size(-2)

    # Copy the in-bounds sub-area of the crop region into the result tensor.
    if h > 0 and w > 0:
        src = tensor.narrow(-2, sy1, h).narrow(-1, sx1, w)
        dst = result.narrow(-2, dy1, h).narrow(-1, dx1, w)
        dst.copy_(src)

    return result


def flip(tensor, horizontal=False, vertical=False):
    """Flip the image.

    Args:
        tensor (Tensor): The image tensor to be flipped.
        horizontal: Flip horizontally.
        vertical: Flip vertically.

    Returns:
        Tensor: The flipped image tensor.
    """
    if horizontal == True:
        tensor = tensor.flip(-1)
    if vertical == True:
        tensor = tensor.flip(-2)
    return tensor


def affine(tensor, matrix):
    """Apply an affine transformation to the image.

    Args:
        tensor (Tensor): The image tensor to be warped.
        matrix (Tensor): The 2x3 affine transformation matrix.

    Returns:
        Tensor: The warped image.
    """
    is_unbatched = tensor.ndimension() == 3
    if is_unbatched:
        tensor = tensor.unsqueeze(0)
    warped = warp_affine(tensor, matrix, tensor.size()[-2:])
    if is_unbatched:
        warped = warped.squeeze(0)
    return warped


def rotate(tensor, degrees):
    """Rotate the image anti-clockwise about the centre.

    Args:
        tensor (Tensor): The image tensor to be rotated.
        degrees (float): The angle through which to rotate.

    Returns:
        Tensor: The rotated image tensor.
    """
    rads = math.radians(degrees)
    h, w = tensor.size()[-2:]
    c = math.cos(rads)
    s = math.sin(rads)
    x = (w - 1) / 2
    y = (h - 1) / 2
    # Transformation matrix for clockwise rotation about the centre of the image.
    matrix = torch.tensor([
        [ c, s, -c * x - s * y + x],
        [-s, c,  s * x - c * y + y],
    ], dtype=torch.float32, device=tensor.device)
    return affine(tensor, matrix)
