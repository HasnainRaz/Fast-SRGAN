from omegaconf import OmegaConf
import torch

from dataloader import SRDataset
from trainer import Trainer


def main():
    config = OmegaConf.load("configs/config.yaml")
    train_dataset = SRDataset(
        config.data.image_dir, config.data.lr_image_size, config.data.scale_factor
    )
    pretrain_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=config.training.batch_size,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.training.batch_size,
    )
    trainer = Trainer(config)
    trainer.pretrain(pretrain_dataloader)
    trainer.train(train_dataloader)


if __name__ == "__main__":
    main()
