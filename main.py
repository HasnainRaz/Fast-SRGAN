from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, DataLoader
from dataloader import SRDataset
from trainer import Trainer


def main():
    config = OmegaConf.load("configs/config.yaml")
    train_dataset = SRDataset(
        config.data.image_dir, config.data.lr_image_size, config.data.scale_factor
    )
    pretrain_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=config.training.pretrain_iterations * config.training.batch_size
    )
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=config.training.iterations * config.training.batch_size
    )
    pretrain_dataloader = DataLoader(
        train_dataset,
        sampler=pretrain_sampler,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=config.training.batch_size,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.training.batch_size,
    )
    trainer = Trainer(config)
    trainer.pretrain(pretrain_dataloader)
    trainer.train(train_dataloader)


if __name__ == "__main__":
    main()
