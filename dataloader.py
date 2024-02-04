import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2


class SRDataset(Dataset):
    def __init__(self, image_dir, lr_image_size, scale_factor=4):
        self.image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.lr_image_size = lr_image_size
        self.hr_image_size = lr_image_size * scale_factor
        self.cropper = A.RandomCrop(width=self.hr_image_size, height=self.hr_image_size, p=1.0)
        self.resize = A.Resize(width=self.lr_image_size, height=self.lr_image_size, p=1, interpolation=cv2.INTER_CUBIC)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        hr_image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        hr_image = self.cropper(image=hr_image)["image"]
        lr_image = self.resize(image=hr_image)["image"]
        hr_image = (torch.from_numpy(hr_image).permute(2, 0, 1).float() / 127.5) - 1.0
        lr_image = (torch.from_numpy(lr_image).permute(2, 0, 1).float() / 127.5) - 1.0
        return lr_image, hr_image
