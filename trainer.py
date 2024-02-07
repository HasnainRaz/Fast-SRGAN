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

    def pretrain(self, train_dataloader):
        fixed_lr_images, fixed_hr_images = None, None
        for fixed_lr_images, fixed_hr_images in train_dataloader:
            break
        if fixed_hr_images is None or fixed_lr_images is None:
            raise ValueError("Images are None, please check the dataloader")
        fixed_lr_images, fixed_hr_images = fixed_lr_images.to(
            self.config.training.device
        ), fixed_hr_images.to(self.config.training.device)
        upsampled_images = torch.nn.functional.upsample(
            (fixed_lr_images.to("cpu") + 1.0) / 2.0, scale_factor=4, mode="bicubic"
        ).to(self.config.training.device)
        self.writer.add_images(
            "Images/Pretrain/HighRes", (fixed_hr_images + 1.0) / 2.0, global_step=0
        )
        self.writer.add_images(
            "Images/Pretrain/Upsampled", upsampled_images, global_step=0
        )
        self.writer.add_scalar(
            "SSIM/Pretrain/Upsampled",
            structural_similarity_index_measure(
                upsampled_images, (fixed_hr_images + 1.0) / 2.0, data_range=1.0
            ),
            global_step=0,
        )
        self.writer.add_scalar(
            "PSNR/Pretrain/Upsampled",
            peak_signal_noise_ratio(
                upsampled_images, (fixed_hr_images + 1.0) / 2.0, data_range=1.0
            ),
            global_step=0,
        )
        for epoch in tqdm(range(self.config.training.pretrain_epochs)):
            self.generator.train()
            for step, (lr_images, hr_images) in tqdm(enumerate(train_dataloader)):
                lr_images, hr_images = lr_images.to(
                    self.config.training.device, non_blocking=True
                ), hr_images.to(self.config.training.device, non_blocking=True)
                self.optim_generator.zero_grad()
                fake_hr_images = self.generator(lr_images)
                loss = self.l1_loss(fake_hr_images, hr_images)
                loss.backward()
                self.optim_generator.step()
                self.writer.add_scalar(
                    "Loss/Generator/Pretrain",
                    loss,
                    global_step=len(train_dataloader) * epoch + step,
                )
            self.generator.eval()
            with torch.no_grad():
                fake_hr_images = self.generator(fixed_lr_images)
            self.writer.add_scalar(
                "SSIM/Pretrian/Generated",
                structural_similarity_index_measure(
                    (fake_hr_images + 1.0) / 2.0,
                    (fixed_hr_images + 1.0) / 2.0,
                    data_range=1.0,
                ),
                global_step=epoch,
            )
            self.writer.add_scalar(
                "PSNR/Pretrain/Generated",
                peak_signal_noise_ratio(
                    (fake_hr_images + 1.0) / 2.0,
                    (fixed_hr_images + 1.0) / 2.0,
                    data_range=1.0,
                ),
                global_step=epoch,
            )
            self.writer.add_images(
                "Images/Pretrain/Generated",
                (fake_hr_images + 1.0) / 2.0,
                global_step=epoch,
            )
        torch.save(self.generator.state_dict(), f"runs/pretrain.pt")

    def train(self, train_dataloader):
        fixed_lr_images = torch.zeros(
            (
                1,
                3,
                self.config.data.lr_image_size,
                1,
                3,
                self.config.data.lr_image_size,
            ),
            device=self.config.training.device,
        )
        fixed_hr_images = torch.zeros(
            (
                1,
                3,
                self.config.data.lr_image_size * self.config.data.scale_factor,
                self.config.data.lr_image_size * self.config.data.scale_factor,
            ),
            device=self.config.training.device,
        )
        for fixed_lr_images, fixed_hr_images in train_dataloader:
            break
        fixed_lr_images, fixed_hr_images = fixed_lr_images.to(
            self.config.training.device
        ), fixed_hr_images.to(self.config.training.device)
        real_labels = torch.ones(
            (self.config.training.batch_size, 1), device=self.config.training.device
        )
        fake_labels = torch.zeros(
            (self.config.training.batch_size, 1), device=self.config.training.device
        )
        upsampled_images = torch.nn.functional.upsample(
            (fixed_lr_images.to("cpu") + 1.0) / 2.0, scale_factor=4, mode="bicubic"
        ).to(self.config.training.device)
        self.writer.add_images(
            "Images/GAN/HighRes", (fixed_hr_images + 1.0) / 2.0, global_step=0
        )
        self.writer.add_images("Images/GAN/Upsampled", upsampled_images, global_step=0)
        self.writer.add_scalar(
            "SSIM/GAN/Upsampled",
            structural_similarity_index_measure(
                upsampled_images, (fixed_hr_images + 1.0) / 2.0, data_range=1.0
            ),
            global_step=0,
        )
        self.writer.add_scalar(
            "PSNR/GAN/Upsampled",
            peak_signal_noise_ratio(
                upsampled_images, (fixed_hr_images + 1.0) / 2.0, data_range=1.0
            ),
            global_step=0,
        )
        if osp.exists("runs/pretrain.pt"):
            self.generator.load_state_dict(torch.load("runs/pretrain.pt"))
        for epoch in tqdm(range(self.config.training.epochs)):
            self.generator.train()
            self.discriminator.train()
            for step, (lr_image, hr_image) in tqdm(enumerate(train_dataloader)):
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
                    global_step=len(train_dataloader) * epoch + step,
                )
                # Get the discriminator loss on generated_images:
                fake_hr_images = self.generator(lr_image).detach()
                y_fake = self.discriminator(fake_hr_images)
                loss_fake = 0.5 * self.loss_fn(y_fake, fake_labels)
                self.writer.add_scalar(
                    "Loss/Discriminator/Fake",
                    loss_fake,
                    global_step=len(train_dataloader) * epoch + step,
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
                    global_step=len(train_dataloader) * epoch + step,
                )
                # Get the content loss for the generator
                fake_features = self.vgg(fake_hr_images)
                real_features = self.vgg(hr_image)
                content_loss = self.l1_loss(fake_features, real_features)
                self.writer.add_scalar(
                    "Loss/Generator/Content",
                    content_loss,
                    global_step=len(train_dataloader) * epoch + step,
                )
                # Train the generator
                generator_loss = 1e-3 * adv_loss + content_loss
                generator_loss.backward()
                self.optim_generator.step()

            self.generator.eval()
            with torch.no_grad():
                generated_sr_image = self.generator(fixed_lr_images)
                self.writer.add_images(
                    "Images/GAN/Generated",
                    (generated_sr_image + 1.0) / 2.0,
                    global_step=epoch,
                )
            self.writer.add_scalar(
                "SSIM/GAN/Generated",
                structural_similarity_index_measure(
                    (generated_sr_image + 1.0) / 2.0,
                    (fixed_hr_images + 1.0) / 2.0,
                    data_range=1.0,
                ),
                global_step=epoch,
            )
            self.writer.add_scalar(
                "PSNR/GAN/Generated",
                peak_signal_noise_ratio(
                    (generated_sr_image + 1.0) / 2.0,
                    (fixed_hr_images + 1.0) / 2.0,
                    data_range=1.0,
                ),
                global_step=epoch,
            )
            self.writer.add_images(
                "Images/GAN/Generated",
                (generated_sr_image + 1.0) / 2.0,
                global_step=epoch,
            )
            save_dir = osp.join("runs", self.config.experiment.name)
            torch.save(
                self.generator.state_dict(),
                osp.join(save_dir, f"generator_epoch_{epoch}.pt"),
            )
            torch.save(
                self.discriminator.state_dict(),
                osp.join(save_dir, f"discriminator_epoch_{epoch}.pt"),
            )
            torch.save(
                self.optim_generator.state_dict(),
                osp.join(save_dir, f"generator_optim_{epoch}.pt"),
            )
            torch.save(
                self.optim_discriminator.state_dict(),
                osp.join(save_dir, f"discriminator_optim_{epoch}.pt"),
            )
