import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class NumpyImagesDataset(Dataset):

    def __init__(self, numpy_paths, lr_image_size, scale_factor):
        self.numpy_paths = numpy_paths
        self.lr_image_size = lr_image_size
        self.hr_image_size = lr_image_size * scale_factor
        self.resize = v2.Resize(
            (self.lr_image_size, self.lr_image_size),
            antialias=True,
            interpolation=v2.InterpolationMode.BICUBIC,
        )

    def __len__(self):
        return len(self.numpy_paths)

    def __getitem__(self, idx):
        image = np.load(self.numpy_paths[idx], mmap_mode="c")
        _, h, w = image.shape
        crop_h, crop_w = random.randint(0, h - self.hr_image_size), random.randint(
            0, w - self.hr_image_size
        )
        hr_image = image[
            :, crop_h : crop_h + self.hr_image_size, crop_w : crop_w + self.hr_image_size
        ]
        hr_image = torch.tensor(hr_image, dtype=torch.float32)
        lr_image = self.resize(hr_image)

        hr_image = (hr_image / 127.5) - 1.0
        lr_image = (lr_image / 127.5) - 1.0
        return lr_image, hr_image
