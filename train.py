from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, DataLoader
from dataloader import DIV2KImage, LMDBDataset
from trainer import Trainer
import lmdb
import cv2
import os
import pickle
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np
import random
from multiprocessing import Manager


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_images_to_lmdb(image_list, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=int(20e9))
    with env.begin(write=True) as txn:
        for image_path in tqdm(image_list, desc="Writing LMDB"):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            value = DIV2KImage(deepcopy(image))
            key = str(os.path.basename(image_path))
            txn.put(key.encode("ascii"), pickle.dumps(value))

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def main():
    config = OmegaConf.load("configs/config.yaml")
    if not os.path.exists(config.data.lmdb_path):
        write_images_to_lmdb([os.path.join(config.data.image_dir, x) for x in os.listdir(config.data.image_dir) if x.endswith(".png")], config.data.lmdb_path)
    g = torch.Generator()
    g.manual_seed(config.experiment.seed)
    seed(config.experiment.seed)
    manager = Manager()
    shared_cache = manager.dict()
    train_dataset = LMDBDataset(
        config.data.lmdb_path, config.data.lr_image_size, config.data.scale_factor, shared_cache
    )
    pretrain_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=config.training.pretrain_iterations * config.training.batch_size, generator=g
    )
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=config.training.iterations * config.training.batch_size, generator=g
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
