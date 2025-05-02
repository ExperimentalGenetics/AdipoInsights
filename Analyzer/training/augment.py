import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import math
import rising.transforms as rtrf
from rising.loading import default_transform_call


def to_tensor_if_needed(x):
    """
    Converts the input (which may be a NumPy array or a PIL Image) into a torch.FloatTensor scaled to [0,1]
    with 3 channels (RGB). If the array is channel-first (e.g. shape (3, H, W)) it is transposed to (H, W, 3).
    """
    if isinstance(x, torch.Tensor):
        if x.ndim == 2:
            return x.unsqueeze(0).expand(3, -1, -1).float()
        elif x.ndim == 3 and x.shape[0] == 1:
            return x.expand(3, -1, -1).float()
        else:
            return x.float()
    else:
        # x is assumed to be a NumPy array
        if x.ndim == 3:
            # If the array is channel-first (e.g. (3, H, W)) rather than (H, W, 3),
            # transpose it.
            if x.shape[0] in [1, 3] and x.shape[-1] != 3:
                x = np.transpose(x, (1, 2, 0))
            # If the array is still not 3-channel (e.g. grayscale), convert to 3-channel.
            if x.ndim == 3 and x.shape[-1] != 3:
                x = np.stack([x.squeeze()] * 3, axis=-1)
        elif x.ndim == 2:
            # Grayscale image; replicate the single channel.
            x = np.stack([x] * 3, axis=-1)
        try:
            pil_img = Image.fromarray(x)
        except Exception as e:
            raise TypeError(f"Cannot convert array with shape {x.shape} and dtype {x.dtype} to PIL Image: {e}")
        return T.ToTensor()(pil_img)

def get_transforms_test_s1():
    """
    Returns a transform function (for S1 data) that accepts keyword arguments.
    It converts the 'data' value to a float tensor (ensuring 3 channels) and normalizes it.
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    def transform_fn(**kwargs):
        x = kwargs["data"]
        x = to_tensor_if_needed(x)
        kwargs["data"] = normalize(x)
        return kwargs

    return transform_fn
    
class ApplyToKey:
    """
    A wrapper to apply a transform to a specific key in a dictionary sample.
    """
    def __init__(self, transform, key='data'):
        self.transform = transform
        self.key = key

    def __call__(self, sample):
        sample[self.key] = self.transform(sample[self.key])
        return sample

class Subsample2D:
    """
    Downsamples a tensor image by taking every nth pixel along H and W.
    """
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, img):
        # Assumes img is a tensor of shape [C, H, W]
        return img[:, ::self.stride, ::self.stride]

class MakeDivisible:
    """
    Pads the spatial dimensions (height and width) so that they become divisible by a given divisor.
    For a tensor of shape [C, H, W], it pads only the right and bottom sides.
    """
    def __init__(self, divisor, dim):
        self.divisor = divisor
        self.dim = dim  # Assume 2 for H and W

    def __call__(self, img):
        s_width = img.shape[-1]
        pad_width = math.ceil(s_width / self.divisor) * self.divisor - s_width
        s_height = img.shape[-2]
        pad_height = math.ceil(s_height / self.divisor) * self.divisor - s_height
        pad = (0, pad_width, 0, pad_height)
        return F.pad(img, pad)

def get_transforms_test_s2():
    """
    Transformation pipeline using torchvision.transforms.
    Assumes each sample is a dictionary with a key "data".
    """
    transform = T.Compose([
        ApplyToKey(T.ToTensor(), key='data'),  # Converts image to [C, H, W] and scales to [0,1]
        ApplyToKey(Subsample2D(stride=2), key='data'),
        ApplyToKey(T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]), key='data'),
        ApplyToKey(MakeDivisible(divisor=32, dim=2), key='data'),
    ])
    return transform
