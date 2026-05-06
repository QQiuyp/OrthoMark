import torch
import torch.nn as nn
import kornia.augmentation as K


class Erase(torch.nn.Module):
    """Random erasing attack"""
    def __init__(self,scale=(.3, .9)):
        super().__init__()
        self.aug = K.RandomErasing(scale, (1,1), p=1.0)

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        image = (image + 1) / 2
        image = self.aug(image)
        image = image * 2 - 1
        return image
