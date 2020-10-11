import os

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch
from dataloader import SuperResolutionDataset
from pl_module import FastSRGAN


@hydra.main(config_path='configs', config_name='super_resolution')
def main(cfg):
    image_paths = sorted([os.path.join(cfg.DATA.IMAGE_DIR, x) for x in os.listdir(cfg.DATA.IMAGE_DIR)])
    num_train = int(len(image_paths) * 0.95)
    train_image_paths = image_paths[:num_train]
    val_image_paths = image_paths[num_train:]
    train_loader = DataLoader(SuperResolutionDataset(train_image_paths, cfg.DATA.HIGH_RES_IMAGE_SIZE),
                              batch_size=cfg.TRAINING.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.DATA.NUM_WORKERS)
    val_loader = DataLoader(SuperResolutionDataset(val_image_paths, cfg.DATA.HIGH_RES_IMAGE_SIZE),
                            batch_size=cfg.TRAINING.BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.DATA.NUM_WORKERS)
    checkpoint_callback = ModelCheckpoint(monitor='generated_ssim', mode='max')
    module = FastSRGAN(cfg.MODEL)
    module.example_input_array = torch.rand((1, 3, 256, 256))
    trainer = Trainer(max_epochs=cfg.TRAINING.EPOCHS, gpus=cfg.TRAINING.GPUS, checkpoint_callback=checkpoint_callback,
                      limit_train_batches=0.2)
    trainer.fit(module, train_loader, val_loader)


if __name__ == '__main__':
    main()
