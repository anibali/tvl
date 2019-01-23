"""Functions for transforming image data stored in PyTorch tensors.

This module is necessary since most of the transformations provided by the `torchvision` package
are applicable for PIL.Image images only. Since tvl may load video frames on the GPU, we want
to be able to take the computation to the data rather than moving the images to and from main
memory.

As an additional benefit, these functions are defined such that they also work in batched mode,
which is especially useful for videos.
"""

import torch
from torch.nn.functional import interpolate


def normalise(tensor, mean, stddev, inplace=False):
    mean = torch.as_tensor(mean, device=tensor.device)[..., :, None, None]
    stddev = torch.as_tensor(stddev, device=tensor.device)[..., :, None, None]

    if inplace:
        tensor.sub_(mean)
    else:
        tensor = tensor.sub(mean)

    tensor.div_(stddev)
    return tensor


def denormalise(tensor, mean, stddev, inplace=False):
    mean = torch.as_tensor(mean, device=tensor.device)[..., :, None, None]
    stddev = torch.as_tensor(stddev, device=tensor.device)[..., :, None, None]

    if inplace:
        return tensor.mul_(stddev).add_(mean)
    else:
        return torch.addcmul(mean, tensor, stddev)


def resize(tensor, size, mode='bilinear'):
    is_unbatched = tensor.ndimension() == 3
    if is_unbatched:
        tensor = tensor.unsqueeze(0)
    resized = interpolate(tensor, size=size, mode=mode)
    if is_unbatched:
        resized = resized.squeeze(0)
    return resized
