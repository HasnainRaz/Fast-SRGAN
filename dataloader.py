import os

import albumentations as albu
import cv2
from torch.utils.data import Dataset
import torch


class ImagePairDataset(Dataset):
    """Data Loader for the SR GAN, that prepares a dataset object for training."""

    def __init__(self, image_dir, hr_image_size):
        """
        Initializes the dataset.
        Args:
            image_dir: The path to the directory containing high resolution images.
            hr_image_size: Integer, the crop size of the images to train on (High
                           resolution images will be cropped to this width and height).
        Returns:
            The dataset object.
        """
        self.image_paths = [os.path.join(image_dir, x)
                            for x in os.listdir(image_dir)]
        self.image_size = hr_image_size
        self.cropper = albu.RandomCrop(
            height=hr_image_size, width=hr_image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        high_res = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)
        height, width = high_res.shape[:2]

        if height < self.image_size or width < self.image_size:
            scale_factor = min(height // self.image_size,
                               width // self.image_size)
            new_height, new_width = int(
                height / scale_factor), int(width / scale_factor)
            high_res = cv2.resize(
                high_res, (new_width, new_height), interpolation=cv2.INTER_AREA)

        high_res = self.cropper(image=high_res)['image']
        low_res = cv2.resize(
            high_res, (self.image_size // 4, self.image_size // 4), interpolation=cv2.INTER_LINEAR)
        high_res = (high_res / 127.5) - 1.0
        low_res = (low_res / 127.5) - 1.0
        high_res = torch.from_numpy(high_res).permute(2, 0, 1).float()
        low_res = torch.from_numpy(low_res).permute(2, 0, 1).float()
        return low_res, high_res
