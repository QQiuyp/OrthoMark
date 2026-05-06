import torch
import torch.nn as nn
import kornia.augmentation as K


class Elastic(torch.nn.Module):
    """Elastic transformation attack"""
    def __init__(self,alpha=(15.0, 15.0)):
        super().__init__()
        self.aug = K.RandomElasticTransform(
            kernel_size=(63, 63),
            sigma=(10.0, 10.0),
            alpha=alpha,
            align_corners=True,
            p=1.0,
        )

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        return self.aug(image)
