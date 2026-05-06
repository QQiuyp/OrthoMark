import torch.nn as nn
import random
import torch


def get_random_int(int_range: [int]):
    return random.randint(int_range[0], int_range[1])


class Combined(nn.Module):
    def __init__(self, list=None):
        super(Combined, self).__init__()
        self.list = list

    def forward(self, image_and_cover):
        id = get_random_int([0, len(self.list) - 1])
        return self.list[id](image_and_cover)


class PCombined(nn.Module):
    def __init__(self, list=None):
        super(PCombined, self).__init__()
        self.list = list

    def forward(self, image_and_cover):
        image, cover = image_and_cover
        no_image_list = []
        for id in range(len(self.list)):
            noise_image = self.list[id]([image, cover])
            no_image_list.append(noise_image)
        return no_image_list
