import torch.nn as nn
from kornia.filters import MedianBlur


class MF(nn.Module):
    """Median Filter attack"""
    def __init__(self, kernel=7):
        super(MF, self).__init__()
        self.middle_filter = MedianBlur((kernel, kernel))

    def forward(self, images_clean):
        images, clean_image = images_clean
        return self.middle_filter(images)
