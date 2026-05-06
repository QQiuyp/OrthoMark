import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia



class RC(nn.Module):
    def __init__(self, min_crop_size=13, max_crop_size=128, output_size=(128, 128)):
        super(RC, self).__init__()
        assert output_size[0] == output_size[1]
        self.min_crop_size = int(min_crop_size)
        self.max_crop_size = int(max_crop_size)
        self.output_size = tuple(output_size)
        self.resize = kornia.geometry.transform.Resize(self.output_size)

    def forward(self, image_and_cover):
        image, _ = image_and_cover         
        B, C, H, W = image.shape
        s = int(torch.randint(self.min_crop_size, self.max_crop_size + 1, (1,), device=image.device).item())
        s_eff = int(min(s, H, W))
        crop = K.RandomCrop(size=(s_eff, s_eff), p=1.0, keepdim=False, same_on_batch=False)
        cropped = crop(image)
        return self.resize(cropped)


class Resize(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, image_and_cover):
        image, cover = image_and_cover
        B, C, H, W = image.shape
        new_h = max(1, int(round(H * self.alpha)))
        new_w = max(1, int(round(W * self.alpha)))

        resize_op = kornia.geometry.transform.Resize((new_h, new_w))
        resized = resize_op(image)
        return resized


class Crop(nn.Module):
    def __init__(self, size: float):
        super().__init__()
        self.size = size

    def forward(self, image_and_cover):
        image, _ = image_and_cover         # image: (B,C,H,W)
        B, C, H, W = image.shape

        crop = K.RandomCrop(size=(self.size, self.size), p=1.0, keepdim=False, same_on_batch=False)
        cropped = crop(image)

        return cropped


class AF(nn.Module):
    """Affine transformation (shear and rotation)"""
    def __init__(self, s=(0.0, 0.0), r=(0.0, 0.0)):
        super(AF, self).__init__()
        self.affine = K.RandomAffine(
            degrees=r,
            shear=s,
            p=1.0,  # always apply
            same_on_batch=True,
        )

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover  # ignore cover_image
        image = (image + 1) / 2
        return self.affine(image) * 2 - 1
