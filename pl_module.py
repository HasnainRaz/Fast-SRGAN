import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as metrics
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from losses import RALoss
from model import FastGenerator, Discriminator, MobileNetEncoder, VGGEncoder


class FastSRGAN(pl.LightningModule):
    def __init__(self, hparams):
        super(FastSRGAN, self).__init__()
        self.hparams = hparams
        self.generator = FastGenerator(hparams)
        self.discriminator = Discriminator(hparams)
        self.loss_fn = RALoss()
        if hparams.FEATURE_EXTRACTOR == 'mobilenet':
            self.feature_extractor = MobileNetEncoder()
        elif hparams.FEATURE_EXTRACTOR == 'vgg':
            self.feature_extractor = VGGEncoder()
        else:
            raise ValueError(
                "Feature extractor can either be 'mobilenet' or 'vgg', please set the appropriate name in the config")
        self.x_cache, self.y_cache, self.z_cache = None, None, None
        for p in self.feature_extractor.parameters():
            p.trainable = False

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        self.feature_extractor.eval()
        z = self.generator(x)
        if optimizer_idx == 0:
            for p in self.discriminator.parameters():
                p.trainable = False
            for p in self.generator.parameters():
                p.trainable = True

            true_prediction = self.discriminator(y)
            fake_prediction = self.discriminator(z)
            fake_features = self.feature_extractor((z + 1) / 2.0)
            true_features = self.feature_extractor((y + 1) / 2.0)

            content_loss = F.mse_loss(fake_features, true_features)
            adversarial_loss = self.loss_fn(fake_prediction, true_prediction)
            mse_loss = F.mse_loss(z, y)
            adv_loss = self.hparams.GENERATOR.ADVERSARIAL_WEIGHT * adversarial_loss
            content_loss = self.hparams.GENERATOR.CONTENT_WEIGHT * content_loss
            mse_loss = self.hparams.GENERATOR.MSE_WEIGHT * mse_loss
            g_loss = adv_loss + content_loss + mse_loss

            if batch_idx % 10 == 0:
                self.logger.experiment.add_scalar('Loss/Generator', g_loss, self.global_step)

            return {'loss': g_loss}

        if optimizer_idx == 1:
            for p in self.discriminator.parameters():
                p.trainable = True
            for p in self.generator.parameters():
                p.trainable = False

            true_prediction = self.discriminator(y).mean()
            fake_prediction = self.discriminator(z.detach()).mean()
            d_loss = self.loss_fn(true_prediction, fake_prediction)
            if batch_idx % 10 == 0:
                self.logger.experiment.add_scalar('Loss/Discriminator', d_loss, self.global_step)

            return {'loss': d_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.x_cache, self.y_cache = x, y
        self.z_cache = self(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        ssim_bilinear = metrics.ssim(x, y)
        ssim_generated = metrics.ssim(self.z_cache, y)
        psnr_bilinear = metrics.psnr(x, y)
        psnr_generated = metrics.psnr(self.z_cache, y)

        return {'ssim_bilinear': ssim_bilinear,
                'ssim_generated': ssim_generated,
                'psnr_bilinear': psnr_bilinear,
                'psnr_generated': psnr_generated}

    def validation_epoch_end(self, data):
        avg_bilinear_ssim = torch.stack([x['ssim_bilinear'] for x in data]).mean()
        avg_generated_ssim = torch.stack([x['ssim_generated'] for x in data]).mean()
        avg_bilinear_psnr = torch.stack([x['psnr_bilinear'] for x in data]).mean()
        avg_generated_psnr = torch.stack([x['psnr_generated'] for x in data]).mean()

        self.logger.experiment.add_image('Generated',
                                         make_grid(self.z_cache,
                                                   scale_each=True,
                                                   normalize=True),
                                         self.current_epoch)
        self.logger.experiment.add_image('LowRes',
                                         make_grid(self.x_cache,
                                                   scale_each=True,
                                                   normalize=True),
                                         self.current_epoch)
        self.logger.experiment.add_image('HighRes',
                                         make_grid(self.y_cache,
                                                   scale_each=True,
                                                   normalize=True),
                                         self.current_epoch)
        self.logger.experiment.flush()

        self.log('SSIM/Generated', avg_generated_ssim, logger=True)
        self.log('SSIM/Bilinear', avg_bilinear_ssim, logger=True)
        self.log('PSNR/Generated', avg_generated_psnr, logger=True)
        self.log('PSNR/Bilinear', avg_bilinear_psnr, logger=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.hparams.GENERATOR.LEARNING_RATE,
                                 weight_decay=self.hparams.GENERATOR.WEIGHT_DECAY)
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.hparams.DISCRIMINATOR.LEARNING_RATE,
                                 weight_decay=self.hparams.DISCRIMINATOR.WEIGHT_DECAY)

        return [opt_g, opt_d], []
