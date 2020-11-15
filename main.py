import os

import hydra
from torch.utils.data import DataLoader

from dataloader import SuperResolutionDataset
from trainer import FastSRGAN


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
    module = FastSRGAN(cfg.MODEL)
    module.train(train_loader, val_loader, epochs=cfg.TRAINING.EPOCHS)


if __name__ == '__main__':
    main()
