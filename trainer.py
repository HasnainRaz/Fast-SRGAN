import os.path as osp

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from model import VGG19, Discriminator, Generator


class Trainer:
    fixed_lr_images = torch.tensor([])
    fixed_hr_images = torch.tensor([])

    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(log_dir=osp.join("runs", config.experiment.name))
        self.generator = Generator(config=config.generator)
        self.generator.to(self.config.training.device)
        self.discriminator = Discriminator(config=config.discriminator)
        self.discriminator.to(self.config.training.device)
        self.perceptual_network = VGG19().to(self.config.training.device)
        if config.training.compiled and torch.cuda.is_available():
            self.generator = torch.compile(self.generator, mode="max-autotune")
            self.discriminator = torch.compile(self.discriminator, mode="max-autotune")
            self.perceptual_network = torch.compile(self.perceptual_network, mode="max-autotune")

        # The VGG just provides features, no gradient needed
        self.perceptual_network.eval()
        for p in self.perceptual_network.parameters():
            p.requires_grad = False

        self.optim_generator = torch.optim.AdamW(
            self.generator.parameters(), lr=self.config.training.generator_lr, fused=True
        )
        self.optim_discriminator = torch.optim.AdamW(
            self.discriminator.parameters(), lr=self.config.training.discriminator_lr, fused=True
        )

        # Loss function for the adversarial players
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # Loss function for the content loss
        self.l1_loss = torch.nn.SmoothL1Loss()

        # Metrics for our optimization
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(
            config.training.device
        )
        self.psnr = PeakSignalNoiseRatio(data_range=1.0, reduction="none").to(
            config.training.device
        )

    @torch.no_grad
    def _calculate_metrics_over_dataset(self, dataloader, phase, step):
        self.generator.eval()
        self.ssim.reset()
        self.psnr.reset()
        for lr_images, hr_images in tqdm(
            dataloader, desc="Calculating metrics", total=len(dataloader)
        ):
            lr_images, hr_images = lr_images.to(
                self.config.training.device, non_blocking=True
            ), hr_images.to(self.config.training.device, non_blocking=True)
            sr_images = (1.0 + self.generator(lr_images)) / 2.0
            self.ssim.update(sr_images, (1.0 + hr_images) / 2.0)
            self.psnr.update(sr_images, (1.0 + hr_images) / 2.0)
        self.writer.add_scalar(f"{phase}/SSIM", self.ssim.compute().mean(), global_step=step)
        self.writer.add_scalar(f"{phase}/PSNR", self.psnr.compute().mean(), global_step=step)
        self.writer.flush()

    def _log_fixed_images(self, phase):
        Trainer.fixed_hr_images = Trainer.fixed_hr_images.to(self.config.training.device)
        Trainer.fixed_lr_images = Trainer.fixed_lr_images.to(self.config.training.device)
        upsampled_images = torch.nn.functional.interpolate(
            Trainer.fixed_lr_images.cpu(), scale_factor=4, mode="bicubic", antialias=True
        ).to(self.config.training.device)
        self.writer.add_images(f"{phase}/HighRes", Trainer.fixed_hr_images, global_step=0)
        self.writer.add_images(f"{phase}/Bicubic", upsampled_images, global_step=0)

    @classmethod
    def _pre_train_setup(cls, dataloader):
        if cls.fixed_lr_images.ndim == 1:
            for fixed_lr_images, fixed_hr_images in dataloader:
                cls.fixed_lr_images = (fixed_lr_images + 1.0) / 2.0
                cls.fixed_hr_images = (fixed_hr_images + 1.0) / 2.0
                cls.images_are_set = True
                break

    def pretrain(self, train_dataloader, val_dataloader):
        if osp.exists("runs/pretrain.pt"):
            print("Pretrained model found, skipping pretraining")
            self.generator.load_state_dict(torch.load("runs/pretrain.pt")["model"])
            self.optim_generator.load_state_dict(torch.load("runs/pretrain.pt")["optimizer"])
            return
        self._calculate_metrics_over_dataset(val_dataloader, "Pretrain", step=0)
        self._pre_train_setup(val_dataloader)
        self._log_fixed_images("Pretrain")
        step = 0
        for step, (lr_images, hr_images) in tqdm(
            enumerate(train_dataloader, start=1),
            desc="Pretraining Generator",
            total=len(train_dataloader),
        ):
            lr_images, hr_images = lr_images.to(
                self.config.training.device, non_blocking=True
            ), hr_images.to(self.config.training.device, non_blocking=True)
            self.optim_generator.zero_grad(set_to_none=True)
            fake_hr_images = self.generator(lr_images)
            gen_loss = self.l1_loss(fake_hr_images, hr_images)
            gen_loss.backward()
            self.optim_generator.step()

            if step % self.config.training.log_iter == 0:
                self.writer.add_scalar(
                    "Pretrain/Generator/Loss",
                    gen_loss,
                    global_step=step,
                )
            if step % self.config.training.checkpoint_iter == 0:
                self.generator.eval()
                with torch.no_grad():
                    fake_hr_images = (1.0 + self.generator(2.0 * self.fixed_lr_images - 1.0)) / 2.0
                self.writer.add_images(
                    "Pretrain/Generated",
                    fake_hr_images,
                    global_step=step,
                )
                self._calculate_metrics_over_dataset(val_dataloader, "Pretrain", step)
                self.generator.train()

        torch.save(
            {"model": self.generator.state_dict(), "optimizer": self.optim_generator.state_dict()},
            f"runs/pretrain_generator.pt",
        )
        torch.save(
            {
                "model": self.discriminator.state_dict(),
                "optimizer": self.optim_discriminator.state_dict(),
            },
            f"runs/pretrain_discriminator.pt",
        )

    def save_checkpoints(self, step):
        save_dir = osp.join("runs", self.config.experiment.name)
        torch.save(self.generator.state_dict(), osp.join(save_dir, f"generator_epoch_{step}.pt"))
        torch.save(
            self.discriminator.state_dict(), osp.join(save_dir, f"discriminator_epoch_{step}.pt")
        )
        torch.save(
            self.optim_generator.state_dict(),
            osp.join(save_dir, f"generator_optim_epoch_{step}.pt"),
        )
        torch.save(
            self.optim_discriminator.state_dict(),
            osp.join(save_dir, f"discriminator_optim_epoch_{step}.pt"),
        )

    def train(self, train_dataloader, val_dataloader):
        self._calculate_metrics_over_dataset(val_dataloader, "GAN", step=0)
        if Trainer.fixed_lr_images is None:
            self._pre_train_setup(train_dataloader)
            self._log_fixed_images("GAN")
        self.generator.train()
        self.discriminator.train()
        for step, (lr_images, hr_images) in tqdm(
            enumerate(train_dataloader, start=1), desc="GAN Training", total=len(train_dataloader)
        ):
            lr_images, hr_images = lr_images.to(
                self.config.training.device, non_blocking=True
            ), hr_images.to(self.config.training.device, non_blocking=True)
            self.optim_discriminator.zero_grad(set_to_none=True)
            y_real = self.discriminator(hr_images)
            sr_images = self.generator(lr_images).detach()
            y_fake = self.discriminator(sr_images)
            real_labels = 0.3 * torch.rand_like(y_real) + 0.8
            fake_labels = 0.3 * torch.rand_like(y_fake)
            loss_real = self.loss_fn(y_real, real_labels.to(self.config.training.device))
            loss_fake = self.loss_fn(y_fake, fake_labels.to(self.config.training.device))
            discriminator_loss = 0.5 * loss_real + 0.5 * loss_fake
            discriminator_loss.backward()
            self.optim_discriminator.step()

            # Get the adv loss for the generator
            self.optim_generator.zero_grad(set_to_none=True)
            sr_images = self.generator(lr_images)
            y_fake = self.discriminator(sr_images)
            real_labels = 0.3 * torch.rand_like(y_fake) + 0.7
            adv_loss = 1e-1 * self.loss_fn(y_fake, real_labels.to(self.config.training.device))
            # Get the content loss for the generator
            fake_features = self.perceptual_network(sr_images)
            real_features = self.perceptual_network(hr_images)
            content_loss = self.l1_loss(fake_features, real_features)
            # Train the generator
            generator_loss = 0.5 * adv_loss + 0.5 * content_loss
            generator_loss.backward()
            self.optim_generator.step()

            if step % self.config.training.log_iter == 0:
                self.writer.add_scalar(
                    "Loss/Discriminator/Real",
                    loss_real,
                    global_step=step,
                )
                self.writer.add_scalar(
                    "Loss/Discriminator/Fake",
                    loss_fake,
                    global_step=step,
                )
                self.writer.add_scalar(
                    "Loss/Generator/Adversarial",
                    adv_loss,
                    global_step=step,
                )
                self.writer.add_scalar(
                    "Loss/Generator/Content",
                    content_loss,
                    global_step=step,
                )

            if step % self.config.training.checkpoint_iter == 0:
                self.generator.eval()
                with torch.no_grad():
                    generated_sr_image = (
                        1.0 + self.generator(2 * self.fixed_lr_images - 1.0)
                    ) / 2.0
                    self.writer.add_images(
                        "GAN/Generated",
                        generated_sr_image,
                        global_step=step,
                    )
                    self._calculate_metrics_over_dataset(val_dataloader, "GAN", step=step)
                self.save_checkpoints(step)
                self.generator.train()
