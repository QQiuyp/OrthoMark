import torch
import torch.nn as nn
import numpy as np


class Dropout(nn.Module):
    """
    Drops random pixels from the noised image and substitues them with the pixels from the cover image
    """
    def __init__(self, prob=0.4):
        super(Dropout, self).__init__()
        self.prob = prob

    def forward(self, images_clean):
        noised_image, clean_image = images_clean
        mask_percent = self.prob
        mask = np.random.choice([0.0, 1.0], noised_image.shape[2:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        mask_tensor.unsqueeze_(0)
        mask_tensor.unsqueeze_(0)
        mask_tensor = mask_tensor.expand_as(noised_image)
        noised_image = noised_image * mask_tensor + clean_image * (1 - mask_tensor)
        return noised_image
