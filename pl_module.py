import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import psnr, ssim
from torchvision.utils import make_grid

from model import Discriminator, FeatureExtractor, Generator


class FastSRGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.generator = Generator(
            hparams.GENERATOR.NUM_FILTERS, hparams.GENERATOR.NUM_BLOCKS)
        self.discriminator = Discriminator(
            hparams.DISCRIMINATOR.NUM_FILTERS, hparams.DISCRIMINATOR.NUM_BLOCKS)
        self.feature_extractor = FeatureExtractor(
            hparams.FEATURE_EXTRACTOR.FEATURE_LAYER)
        self.feature_extractor.eval()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        if optimizer_idx == 0:
            sr = self(x)
            fake_discriminated = self.discriminator(sr)
            hr_features = self.feature_extractor(y)
            sr_features = self.feature_extractor(sr)

            feature_loss = F.mse_loss(
                sr_features, hr_features) * self.hparams.LOSS.FEATURE_WEIGHT
            adversarial_loss = F.binary_cross_entropy_with_logits(
                fake_discriminated, torch.ones_like(fake_discriminated)) * self.hparams.LOSS.ADVERSARIAL_WEIGHT
            pixel_loss = F.mse_loss(sr, y) * self.hparams.LOSS.PIXEL_WEIGHT

            g_loss = feature_loss + adversarial_loss + pixel_loss

            if self.global_step % 100 == 0:
                self.logger.experiment.add_scalar(
                    'Loss/Pixel', pixel_loss, global_step=self.global_step)
                self.logger.experiment.add_scalar(
                    'Loss/Feature', feature_loss, global_step=self.global_step)
                self.logger.experiment.add_scalar(
                    'Loss/Adversarial', adversarial_loss, global_step=self.global_step)
                self.logger.experiment.add_scalar(
                    'Loss/Generator', g_loss, global_step=self.global_step)
                self.logger.experiment.add_scalar(
                    'PSNR/SR', psnr(sr, y), global_step=self.global_step)
                self.logger.experiment.add_scalar('PSNR/HR', psnr(F.interpolate(
                    x, scale_factor=4, mode='bilinear', align_corners=False), y), global_step=self.global_step)
                self.logger.experiment.add_scalar(
                    'SSIM/SR', ssim(sr, y), global_step=self.global_step)
                self.logger.experiment.add_scalar('SSIM/HR', ssim(F.interpolate(
                    x, scale_factor=4, mode='bilinear', align_corners=False), y), global_step=self.global_step)
                self.logger.experiment.add_image(
                    'Images/HR', make_grid(y, normalize=True, scale_each=True), global_step=self.global_step)
                self.logger.experiment.add_image('Images/LR', make_grid(F.interpolate(
                    x, scale_factor=4, mode='bilinear', align_corners=False), normalize=True, scale_each=True),
                    global_step=self.global_step)
                self.logger.experiment.add_image(
                    'Images/SR', make_grid(sr, normalize=True, scale_each=True), global_step=self.global_step)
                self.logger.experiment.flush()
            return g_loss

        if optimizer_idx == 1:
            sr = self(x).detach()
            real_discriminated = self.discriminator(y)
            fake_discriminated = self.discriminator(sr)

            real_loss = F.binary_cross_entropy_with_logits(
                real_discriminated, torch.ones_like(real_discriminated))
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_discriminated, torch.zeros_like(fake_discriminated))

            d_loss = (real_loss + fake_loss) / 2
            if self.global_step % 100 == 0:
                self.logger.experiment.add_scalar(
                    'Loss/Discriminator', d_loss, global_step=self.global_step)
                self.logger.experiment.flush()
            return d_loss

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.generator.parameters(),
                             lr=self.hparams.GENERATOR.LEARNING_RATE),
            torch.optim.Adam(self.discriminator.parameters(),
                             lr=self.hparams.DISCRIMINATOR.LEARNING_RATE)
        ]
