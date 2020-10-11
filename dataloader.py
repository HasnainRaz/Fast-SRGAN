import random

import cv2
import torch
from torch.utils.data import Dataset


class SuperResolutionDataset(Dataset):
    """Data Set object that reads high resolution images and
       generates training data pairs of high/low res images."""

    def __init__(self, image_paths, hr_image_size):
        """
                Initializes the dataloader.
        Args:
            image_paths: List of paths to the high resolution images.
            hr_image_size: Integer, the crop size of the images to train on (High
                           resolution images will be cropped to this width and height).
        Returns:
            None
        """
        self.image_paths = image_paths
        self.image_size = hr_image_size

    def __len__(self):
        return len(self.image_paths)

    def rescale_image(self, image):
        h, w = image.shape[:2]
        scale_factor = min(h, w) / self.image_size
        if scale_factor < 1:
            if h < w:
                h, w = self.image_size, w * scale_factor
            else:
                h, w = h * scale_factor, self.image_size
            h, w = int(h + 0.5), int(w + 0.5)
            image = cv2.resize(image, (w, h), cv2.INTER_CUBIC)
        return image

    def random_crop(self, image):
        h, w = image.shape[:2]
        x1 = random.randint(0, w - self.image_size)
        y1 = random.randint(0, h - self.image_size)
        x2 = x1 + self.image_size
        y2 = y1 + self.image_size

        return image[y1:y2, x1:x2]

    @staticmethod
    def normalize(high_res, low_res):
        high_res = (high_res / 255.0) * 2 - 1
        low_res = (low_res / 255.0) * 2 - 1

        return high_res, low_res

    def __getitem__(self, idx):
        """
        Returns a single high/low res pair of images
        Args:
            idx: int, the index of the image to sample
        Return:
            high_res: Tensor, high resolution image tensor.
            low_res: Tensor, low resolution image tensor.
        """
        high_res = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)
        high_res = self.rescale_image(high_res)
        high_res = self.random_crop(high_res)
        low_res = cv2.resize(high_res, (self.image_size // 4, self.image_size // 4), cv2.INTER_LINEAR)
        high_res, low_res = self.normalize(high_res, low_res)
        high_res = torch.from_numpy(high_res).permute(2, 0, 1)
        low_res = torch.from_numpy(low_res).permute(2, 0, 1)
        return low_res.float(), high_res.float()
