import os
import pickle

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class DIV2KImage:

    def __init__(self, image):
        self.channels = 3
        self.size = image.shape[:2]
        self.image = image.tobytes()

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return torch.from_numpy(image.copy().reshape(*self.size, 3)).permute(2, 0, 1)


class LMDBDataset(Dataset):

    def __init__(self, db_path, lr_image_size, scale_factor):
        self.db_path = db_path
        self.lr_image_size = lr_image_size
        self.hr_image_size = lr_image_size * scale_factor
        self.cropper = v2.RandomCrop((self.hr_image_size, self.hr_image_size))
        self.resize = v2.Resize(
            (self.lr_image_size, self.lr_image_size),
            antialias=True,
            interpolation=v2.InterpolationMode.BICUBIC,
        )
        env = lmdb.open(
            db_path,
            subdir=os.path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin() as txn:
            self.keys = sorted(list(txn.cursor().iternext(values=False)))
        env.close()
        self.env = None
        self.txn = None

    def _init_db(self):
        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()
        file_name = self.keys[idx]
        hr_image = self.txn.get(file_name)
        hr_image = pickle.loads(hr_image).get_image()
        hr_image = self.cropper(hr_image)
        lr_image = self.resize(hr_image)

        hr_image = (hr_image.float() / 127.5) - 1.0
        lr_image = (lr_image.float() / 127.5) - 1.0
        return lr_image, hr_image

