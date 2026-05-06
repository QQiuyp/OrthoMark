import torch
import torch.nn as nn
import kornia.augmentation as K
import torchvision.transforms.functional as F


class Bright(nn.Module):
    """Brightness adjustment attack"""
    def __init__(self, min_factor=None, max_factor=None):
        super(Bright, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("min_factor and max_factor must be provided")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image):
        image, _ = image
        image = (image + 1) / 2
        factor = self.get_random_factor()
        image = F.adjust_brightness(image, factor)
        image = image * 2 - 1
        return image


class Contrast(nn.Module):
    """Contrast adjustment attack"""
    def __init__(self, min_factor=None, max_factor=None):
        super(Contrast, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("min_factor and max_factor must be provided")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image):
        image, _ = image
        image = (image + 1) / 2
        factor = self.get_random_factor()
        image = F.adjust_contrast(image, factor)
        image = image * 2 - 1
        return image


class Saturation(nn.Module):
    """Saturation adjustment attack"""
    def __init__(self, min_factor=None, max_factor=None):
        super(Saturation, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("Factor range must be specified")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image):
        image, _ = image
        image = (image + 1) / 2
        factor = self.get_random_factor()
        image = F.adjust_saturation(image, factor)
        image = image * 2 - 1
        return image


class Hue(nn.Module):
    """Hue adjustment attack"""
    def __init__(self, min_factor=None, max_factor=None):
        super(Hue, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("Factor range must be specified")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image):
        image, _ = image
        image = (image + 1) / 2
        factor = self.get_random_factor()
        image = F.adjust_hue(image, factor)
        image = image * 2 - 1
        return image
