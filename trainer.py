import logging
import os

import pytorch_lightning.metrics.functional as metrics
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from model import FastGenerator, Discriminator, VGGEncoder


class FastSRGAN(object):
    def __init__(self, cfg):
        super(FastSRGAN, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = FastGenerator(cfg).to(self.device)
        self.discriminator = Discriminator(cfg).to(self.device)
        self.feature_extractor = VGGEncoder().to(self.device)
        self.x_cache, self.y_cache, self.z_cache = None, None, None
        for p in self.feature_extractor.parameters():
            p.trainable = False

        self.adversarial_loss = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.content_loss = torch.nn.MSELoss().to(self.device)
        self.logger = SummaryWriter('logs')
        self.imagenet_mean = torch.as_tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        self.imagenet_std = torch.as_tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
        os.makedirs('checkpoints', exist_ok=True)

    def normalize(self, batch_tensor):
        batch_tensor -= self.imagenet_mean
        return batch_tensor / self.imagenet_std

    def pretrain(self, train_loader, optimizer):
        step = 0
        pbar = tqdm(range(self.cfg.GENERATOR.PRETRAIN_EPOCHS))
        for _ in pbar:
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                z = self.generator(x)
                loss = self.content_loss(z, y)
                loss.backward()
                optimizer.step()
                pbar.set_description(f'Pretrain Loss: {loss.item()}')
                step += 1
        torch.save(self.generator.state_dict(), 'checkpoints/pretrained_generator.pt')

    def generator_step(self, x, y):
        self.feature_extractor.eval()
        z = self.generator(x)
        fake_prediction = self.discriminator(z)
        fake_features = self.feature_extractor(self.normalize((z + 1) / 2.0))
        true_features = self.feature_extractor(self.normalize((y + 1) / 2.0)).detach()

        content_loss = self.content_loss(fake_features, true_features)
        adversarial_loss = self.adversarial_loss(fake_prediction,
                                                 (torch.ones(fake_prediction.shape) - torch.rand(
                                                     fake_prediction.shape) * 0.2).type_as(fake_prediction))
        adv_loss = self.cfg.GENERATOR.ADVERSARIAL_WEIGHT * adversarial_loss
        content_loss = self.cfg.GENERATOR.CONTENT_WEIGHT * content_loss
        g_loss = adv_loss + content_loss

        return g_loss, z

    def discriminator_step(self, y, z):
        real_prediction = self.discriminator(y)
        real_loss = self.adversarial_loss(real_prediction, (
                torch.ones(real_prediction.shape) - torch.rand(real_prediction.shape) * 0.2).type_as(
            real_prediction))
        fake_prediction = self.discriminator(z.detach())
        fake_loss = self.adversarial_loss(fake_prediction,
                                          (torch.rand(fake_prediction.shape) * 0.3).type_as(fake_prediction))

        d_loss = fake_loss + real_loss
        return d_loss

    def train(self, train_loader, val_loader, epochs):
        opt_g, opt_d = self.configure_optimizers()
        if self.cfg.GENERATOR.PRETRAIN_EPOCHS:
            self.pretrain(train_loader, opt_g)
        if os.path.isfile('checkpoints/pretrained_generator.pt'):
            logging.info('Found pretrained generator, will train SRGAN with these weights')
            self.generator.load_state_dict(torch.load('checkpoints/pretrained_generator.pt'))
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.generator.train()
            self.discriminator.train()
            gen_total_loss, dis_total_loss = [], []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt_g.zero_grad()
                g_loss, sr_image = self.generator_step(x, y)
                g_loss.backward()
                opt_g.step()

                opt_d.zero_grad()
                d_loss = self.discriminator_step(y, sr_image)
                d_loss.backward()
                opt_d.step()

                gen_total_loss.append(g_loss.item())
                dis_total_loss.append(d_loss.item())

            gen_total_loss = sum(gen_total_loss) / len(gen_total_loss)
            dis_total_loss = sum(dis_total_loss) / len(dis_total_loss)
            pbar.set_description(f'Gen Loss: {gen_total_loss}, Disc Loss: {dis_total_loss}')
            self.log_train_losses(gen_total_loss, dis_total_loss, epoch)

            self.generator.eval()
            self.discriminator.eval()
            gen_total_loss = []
            ssim_generated, ssim_bilinear = [], []
            psnr_generated, psnr_bilinear = [], []
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                g_loss, sr_image = self.generator_step(x, y)
                gen_total_loss.append(g_loss.item())
                x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
                ssim_bilinear.append(metrics.ssim(x, y))
                ssim_generated.append(metrics.ssim(sr_image, y))
                psnr_bilinear.append(metrics.psnr(x, y))
                psnr_generated.append(metrics.psnr(sr_image, y))

            gen_total_loss = sum(gen_total_loss) / len(gen_total_loss)
            self.log_val_loss(gen_total_loss, epoch)
            self.log_metrics(ssim_bilinear, ssim_generated, psnr_bilinear, psnr_generated, epoch)
            self.log_images(x, y, sr_image, epoch)

    def log_train_losses(self, g_loss, d_loss, epoch):
        self.logger.add_scalar('Train/GeneratorLoss', g_loss, epoch)
        self.logger.add_scalar('Train/DiscriminatorLoss', d_loss, epoch)

    def log_val_loss(self, g_loss, epoch):
        self.logger.add_scalar('Val/GeneratorLoss', g_loss, epoch)

    def log_metrics(self, ssim_bilinear, ssim_generated, psnr_bilinear, psnr_generated, epoch):
        avg_bilinear_ssim = sum(ssim_bilinear) / len(ssim_bilinear)
        avg_generated_ssim = sum(ssim_generated) / len(ssim_generated)
        avg_bilinear_psnr = sum(psnr_bilinear) / len(psnr_bilinear)
        avg_generated_psnr = sum(psnr_generated) / len(psnr_generated)

        self.logger.add_scalar('SSIM/Generated', avg_generated_ssim, epoch)
        self.logger.add_scalar('SSIM/Bilinear', avg_bilinear_ssim, epoch)
        self.logger.add_scalar('PSNR/Generated', avg_generated_psnr, epoch)
        self.logger.add_scalar('PSNR/Bilinear', avg_bilinear_psnr, epoch)
        self.logger.flush()

    def log_images(self, low_res, high_res, super_res, epoch):
        self.logger.add_image('Generated',
                              make_grid(super_res,
                                        scale_each=True,
                                        normalize=True),
                              epoch)
        self.logger.add_image('LowRes',
                              make_grid(low_res,
                                        scale_each=True,
                                        normalize=True),
                              epoch)
        self.logger.add_image('HighRes',
                              make_grid(high_res,
                                        scale_each=True,
                                        normalize=True),
                              epoch)
        self.logger.flush()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.cfg.GENERATOR.LEARNING_RATE,
                                 weight_decay=self.cfg.GENERATOR.WEIGHT_DECAY)
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.cfg.DISCRIMINATOR.LEARNING_RATE,
                                 weight_decay=self.cfg.DISCRIMINATOR.WEIGHT_DECAY)

        return opt_g, opt_d
