import numpy as np
import torch
import torch.nn as nn
import kornia.augmentation as K


class GN(torch.nn.Module):
    """Gaussian Noise attack"""
    def __init__(self, std = 0.04):
        super().__init__()
        self.aug = K.RandomGaussianNoise(mean=0., std=std, p=1.)

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        image = (image + 1) / 2
        image = self.aug(image)
        image = image * 2 - 1
        return image
