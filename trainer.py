import os.path as osp


import torch
from tqdm import tqdm
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
from torch.utils.tensorboard.writer import SummaryWriter

from model import Generator, Discriminator, VGG19


class Trainer:

    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(log_dir=osp.join("runs", config.experiment.name))
        self.generator = Generator(config=config.generator)
        self.generator.to(self.config.training.device)
        self.discriminator = Discriminator(config=config.discriminator)
        self.discriminator.to(self.config.training.device)
        self.vgg = VGG19(config=config.vgg).to(self.config.training.device)

        # The VGG just provides features, no gradient needed
        self.vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.optim_generator = torch.optim.AdamW(
            self.generator.parameters(), lr=self.config.training.generator_lr
        )
        self.optim_discriminator = torch.optim.AdamW(
            self.discriminator.parameters(), lr=self.config.training.discriminator_lr
        )

        # Loss function for the adversarial players
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # Loss function for the content loss
        self.l1_loss = torch.nn.SmoothL1Loss()
        # Keep some images in cache for tensorboard
        self.fixed_lr_images, self.fixed_hr_images = None, None


    def _pre_train(self, dataloader, phase):
        if self.fixed_hr_images is None or self.fixed_lr_images is None: 
            for fixed_lr_images, fixed_hr_images in dataloader:
                self.fixed_lr_images = (fixed_lr_images + 1.0) / 2.0
                self.fixed_hr_images = (fixed_hr_images + 1.0) / 2.0
                break
            assert self.fixed_hr_images is not None, "HR images are None"
            assert self.fixed_lr_images is not None, "LR images are None"
            self.fixed_hr_images = self.fixed_hr_images.to(self.config.training.device)
            self.fixed_lr_images = self.fixed_lr_images.to(self.config.training.device)
            upsampled_images = torch.nn.functional.interpolate(self.fixed_lr_images.cpu(), scale_factor=4, mode="bicubic", antialias=True).to(self.config.training.device)
            self.writer.add_images(f"{phase}/HighRes", self.fixed_hr_images, global_step=0)
            self.writer.add_images(f"{phase}/Bicubic", upsampled_images, global_step=0)
            self.writer.add_scalar(f"{phase}/PSNR", peak_signal_noise_ratio(upsampled_images, self.fixed_hr_images, data_range=1.0), global_step=0)
            self.writer.add_scalar(f"{phase}/SSIM", structural_similarity_index_measure(upsampled_images, self.fixed_hr_images, data_range=1.0), global_step=0)

    def pretrain(self, train_dataloader):
        self._pre_train(train_dataloader, "Pretrain")
        for step, (lr_images, hr_images) in tqdm(enumerate(train_dataloader, start=1)):
            lr_images, hr_images = lr_images.to(
                self.config.training.device, non_blocking=True
            ), hr_images.to(self.config.training.device, non_blocking=True)
            self.optim_generator.zero_grad()
            fake_hr_images = self.generator(lr_images)
            loss = self.l1_loss(fake_hr_images, hr_images)
            loss.backward()
            self.optim_generator.step()
            self.writer.add_scalar(
                "Pretrain/Loss",
                loss,
                global_step=step,
            )
            if step % 100 == 0:
                self.generator.eval()
                with torch.no_grad():
                    fake_hr_images = (1.0 + self.generator(2.0*self.fixed_lr_images - 1.0)) / 2.0
                self.writer.add_scalar(
                    "Pretrain/SSIM",
                    structural_similarity_index_measure(
                        fake_hr_images,
                        self.fixed_hr_images, 
                        data_range=1.0,
                    ),
                    global_step=step,
                )
                self.writer.add_scalar(
                    "Pretrain/PSNR",
                    peak_signal_noise_ratio(
                        fake_hr_images,
                        self.fixed_hr_images,
                        data_range=1.0,
                    ),
                    global_step=step,
                )
                self.writer.add_images(
                    "Pretrain/Generated",
                    fake_hr_images,
                    global_step=step,
                )
                self.generator.train()
        torch.save(self.generator.state_dict(), f"runs/pretrain.pt")

    def train(self, train_dataloader):
        self._pre_train(train_dataloader, "GAN")
        if osp.exists("runs/pretrain.pt"):
            self.generator.load_state_dict(torch.load("runs/pretrain.pt"))
        real_labels = torch.ones((self.config.training.batch_size, 1), device=self.config.training.device)
        fake_labels = torch.zeros((self.config.training.batch_size, 1), device=self.config.training.device)
        self.generator.train()
        self.discriminator.train()
        for step, (lr_image, hr_image) in tqdm(enumerate(train_dataloader, start=1)):
            lr_image, hr_image = lr_image.to(
                self.config.training.device, non_blocking=True
            ), hr_image.to(self.config.training.device, non_blocking=True)
            self.optim_discriminator.zero_grad()
            # Get the discriminator loss on real images:
            y_real = self.discriminator(hr_image)
            loss_real = 0.5 * self.loss_fn(y_real, real_labels)
            self.writer.add_scalar(
                "Loss/Discriminator/Real",
                loss_real,
                global_step=step,
            )
            # Get the discriminator loss on generated_images:
            fake_hr_images = self.generator(lr_image).detach()
            y_fake = self.discriminator(fake_hr_images)
            loss_fake = 0.5 * self.loss_fn(y_fake, fake_labels)
            self.writer.add_scalar(
                "Loss/Discriminator/Fake",
                loss_fake,
                global_step=step,
            )
            # Train the discriminator
            discriminator_loss = loss_real + loss_fake
            discriminator_loss.backward()
            self.optim_discriminator.step()

            # Get the adv loss for the generator
            self.optim_generator.zero_grad()
            fake_hr_images = self.generator(lr_image)
            y_fake = self.discriminator(fake_hr_images)
            adv_loss = self.loss_fn(y_fake, real_labels)
            self.writer.add_scalar(
                "Loss/Generator/Adversarial",
                adv_loss,
                global_step=step,
            )
            # Get the content loss for the generator
            fake_features = self.vgg(fake_hr_images)
            real_features = self.vgg(hr_image)
            content_loss = self.l1_loss(fake_features, real_features)
            self.writer.add_scalar(
                "Loss/Generator/Content",
                content_loss,
                global_step=step,
            )
            # Train the generator
            generator_loss = 1e-3 * adv_loss + content_loss
            generator_loss.backward()
            self.optim_generator.step()
            
            if step % self.config.training.log_iter == 0:
                self.generator.eval()
                with torch.no_grad():
                    generated_sr_image = (1.0 + self.generator(2 * self.fixed_lr_images - 1.0)) / 2.0
                    self.writer.add_images(
                        "GAN/Generated",
                        generated_sr_image,
                        global_step=step,
                    )
                self.writer.add_scalar(
                    "GAN/SSIM",
                    structural_similarity_index_measure(
                        generated_sr_image, 
                        self.fixed_hr_images,
                        data_range=1.0,
                    ),
                    global_step=step,
                )
                self.writer.add_scalar(
                    "GAN/PSNR",
                    peak_signal_noise_ratio(
                        generated_sr_image,
                        self.fixed_hr_images,
                        data_range=1.0,
                    ),
                    global_step=step,
                )
                self.generator.train()
                save_dir = osp.join("runs", self.config.experiment.name)
                torch.save(
                    self.generator.state_dict(),
                    osp.join(save_dir, f"generator_epoch_{step}.pt"),
                )
                torch.save(
                    self.discriminator.state_dict(),
                    osp.join(save_dir, f"discriminator_epoch_{step}.pt"),
                )
                torch.save(
                    self.optim_generator.state_dict(),
                    osp.join(save_dir, f"generator_optim_{step}.pt"),
                )
                torch.save(
                    self.optim_discriminator.state_dict(),
                    osp.join(save_dir, f"discriminator_optim_{step}.pt"),
                )
