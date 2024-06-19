import os
import random
from concurrent.futures import ThreadPoolExecutor

import hydra
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from dataloader import NumpyImagesDataset
from trainer import Trainer


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_images_to_numpy_arrays(image_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def _write_image_to_numpy(image_path, numpy_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = np.transpose(image, (2, 0, 1))
        np.save(numpy_path, image)
        pbar.update(1)

    with tqdm(total=len(image_list)) as pbar:
        with ThreadPoolExecutor(max_workers=16) as executor:
            for image_path in image_list:
                file_name = os.path.basename(image_path).replace(".png", "")
                numpy_path = os.path.join(output_dir, file_name)
                executor.submit(_write_image_to_numpy, image_path, numpy_path)


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@hydra.main(version_base="1.1", config_path="configs", config_name="config")
def main(config):
    if not os.path.exists(config.data.numpy_dir):
        write_images_to_numpy_arrays(
            [
                os.path.join(config.data.image_dir, x)
                for x in os.listdir(config.data.image_dir)
                if x.endswith(".png")
            ],
            config.data.numpy_dir,
        )
    g = torch.Generator()
    g.manual_seed(config.experiment.seed)
    seed(config.experiment.seed)

    numpy_files = [
        os.path.join(config.data.numpy_dir, x)
        for x in os.listdir(config.data.numpy_dir)
        if x.endswith(".npy")
    ]
    train_dataset = NumpyImagesDataset(
        numpy_files, config.data.lr_image_size, config.data.scale_factor
    )
    pretrain_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=config.training.pretrain_iterations * config.training.batch_size,
        generator=g,
    )
    train_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=config.training.iterations * config.training.batch_size,
        generator=g,
    )
    val_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        drop_last=True,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    pretrain_dataloader = DataLoader(
        train_dataset,
        sampler=pretrain_sampler,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    trainer = Trainer(config)
    trainer.pretrain(pretrain_dataloader, val_dataloader)
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
