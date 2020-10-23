import os

import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from dataloader import SuperResolutionDataset
from pl_module import FastSRGAN


@hydra.main(config_path='configs', config_name='super_resolution')
def main(cfg):
    train_image_paths = sorted(
        [os.path.join(cfg.DATA.TRAIN_IMAGE_DIR, x) for x in os.listdir(cfg.DATA.TRAIN_IMAGE_DIR)])
    val_image_paths = sorted([os.path.join(cfg.DATA.VAL_IMAGE_DIR, x) for x in os.listdir(cfg.DATA.VAL_IMAGE_DIR)])
    train_loader = DataLoader(SuperResolutionDataset(train_image_paths, cfg.DATA.HIGH_RES_IMAGE_SIZE),
                              batch_size=cfg.TRAINING.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.DATA.NUM_WORKERS)
    val_loader = DataLoader(SuperResolutionDataset(val_image_paths, cfg.DATA.HIGH_RES_IMAGE_SIZE),
                            batch_size=cfg.TRAINING.BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.DATA.NUM_WORKERS)
    checkpoint_callback = ModelCheckpoint(monitor='SSIM/Generated', mode='max')
    early_stop_callback = EarlyStopping(monitor='SSIM/Generated', mode='max', patience=20)
    module = FastSRGAN(cfg.MODEL)
    module.example_input_array = torch.rand((1, 3, 256, 256))
    trainer = Trainer(max_epochs=cfg.TRAINING.EPOCHS,
                      gpus=cfg.TRAINING.GPUS,
                      callbacks=[early_stop_callback],
                      checkpoint_callback=checkpoint_callback,
                      precision=cfg.TRAINING.PRECISION)
    trainer.fit(module, train_loader, val_loader)


if __name__ == '__main__':
    main()
