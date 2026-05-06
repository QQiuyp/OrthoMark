import torch
import torch.nn as nn
import kornia.augmentation as K


class SP(torch.nn.Module):
    """Salt and Pepper noise attack"""
    def __init__(self, amount = 0.1):
        super().__init__()
        self.aug = K.RandomSaltAndPepperNoise(amount=amount, salt_vs_pepper=0.5, p=1.0)

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        image = (image + 1) / 2
        image = self.aug(image)
        image = image * 2 - 1
        return image
