import os

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from torch.utils.data import Dataset


class SRDataset(Dataset):
    def __init__(self, image_dir, lr_image_size, scale_factor=4):
        self.image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.lr_image_size = lr_image_size
        self.hr_image_size = lr_image_size * scale_factor
        self.cropper = v2.RandomCrop((self.hr_image_size, self.hr_image_size))
        self.resize = v2.Resize((self.lr_image_size, self.lr_image_size), antialias=True, interpolation=v2.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        hr_image = read_image(self.image_paths[index], mode=ImageReadMode.RGB)
        hr_image = self.cropper(hr_image)
        lr_image = self.resize(hr_image)
        hr_image = (hr_image.float() / 127.5) - 1.0
        lr_image = (lr_image.float() / 127.5) - 1.0
        return lr_image, hr_image
