from torch import nn
from models.combined import Combined, PCombined
from attacks.identity import Identity
from attacks.median_filter import MF
from attacks.gaussian_noise import GN
from attacks.gaussian_blur import GF
from attacks.salt_pepper import SP
from attacks.dropout import Dropout
from attacks.elastic import Elastic
from attacks.geometric import RC, AF
from attacks.erase import Erase
from attacks.color import Bright, Contrast, Hue, Saturation
from attacks.jpeg import Jpeg, JpegTest, JpegSS, JpegMask, KJpeg


class Noise(nn.Module):
    def __init__(self, layers):
        super(Noise, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.noise = nn.Sequential(*layers)

    def forward(self, image_and_cover):
        noised_image = self.noise(image_and_cover)
        return noised_image
