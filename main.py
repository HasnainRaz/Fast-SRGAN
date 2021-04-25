import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataloader import ImagePairDataset
from pl_module import FastSRGAN


def main():
    hparams = OmegaConf.load('config.yaml')
    image_dir = hparams.DATA.IMAGE_DIR
    hr_image_size = hparams.DATA.HIGH_RES_CROP_SIZE
    train_loader = DataLoader(ImagePairDataset(image_dir=image_dir,
                                               hr_image_size=hr_image_size),
                              batch_size=hparams.TRAINING.BATCH_SIZE,
                              num_workers=hparams.TRAINING.NUM_WORKERS,
                              shuffle=True)

    callback = ModelCheckpoint('weights')
    module = FastSRGAN(hparams=hparams)
    trainer = Trainer(max_epochs=hparams.TRAINING.EPOCHS,
                      gpus=hparams.TRAINING.GPUS,
                      callbacks=[callback],
                      resume_from_checkpoint=hparams.TRAINING.RESUME_FROM_CKPT)
    trainer.fit(module, train_dataloader=train_loader)


if __name__ == '__main__':
    main()
