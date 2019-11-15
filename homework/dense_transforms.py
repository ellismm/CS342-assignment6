# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            args = tuple(np.array([image.width-point[0], point[1]], dtype=point.dtype) for point in args)
        return (image,) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args



# def to_heatmap(im, *dets, device=None, **kwargs):
#     size_map = torch.zeros((2,) + im.shape[1:], device=device)
#     det_map = torch.zeros((len(dets),) + im.shape[1:], device=device)
#     # for i, det in enumerate(dets):
#     #     _draw_detections(det, det_map[i], size_map, **kwargs)
#     return im, det_map, size_map


# class ToHeatmap(object):
#     def __init__(self, radius=2):
#         self.radius = radius

#     def __call__(self, image, *args):
#         return to_heatmap(image, *args, radius=self.radius)