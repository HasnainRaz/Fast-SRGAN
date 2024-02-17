import os.path as osp


import torch
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.utils.tensorboard.writer import SummaryWriter

from model import Generator, PerceptualNetwork, Discriminator 


class Trainer:
    fixed_lr_images = None 
    fixed_hr_images = None 


    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(log_dir=osp.join("runs", config.experiment.name))
        self.generator = Generator(config=config.generator)
        self.generator.to(self.config.training.device)
        self.discriminator = Discriminator(config=config.discriminator)
        self.discriminator.to(self.config.training.device)
        self.perceptual_network = PerceptualNetwork(config=config.perceptual_network).to(self.config.training.device)
        if config.training.compiled and torch.cuda.is_available():
            self.generator = torch.compile(self.generator)
            self.discriminator = torch.compile(self.discriminator)
            self.perceptual_network = torch.compile(self.perceptual_network)
        # The VGG just provides features, no gradient needed
        self.perceptual_network.eval()
        for p in self.perceptual_network.parameters():
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

        # Metrics for our optimization
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(config.training.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0, reduction='none').to(config.training.device)

    
    @torch.no_grad
    def _calculate_metrics_over_dataset(self, dataloader, phase, step):
        self.generator.eval() 
        self.ssim.reset()
        self.psnr.reset()
        for lr_images, hr_images in tqdm(dataloader, desc="Calculating metrics", total=len(dataloader)):
            lr_images, hr_images = lr_images.to(self.config.training.device, non_blocking=True), hr_images.to(self.config.training.device, non_blocking=True)
            sr_images = (1.0+self.generator(lr_images)) / 2.0
            self.ssim.update(sr_images, (1.0+hr_images) / 2.0)
            self.psnr.update(sr_images, (1.0+hr_images) / 2.0)
        self.writer.add_scalar(f"{phase}/SSIM", self.ssim.compute().mean(), global_step=step)
        self.writer.add_scalar(f"{phase}/PSNR", self.psnr.compute().mean(), global_step=step)
        self.writer.flush()



    def _log_fixed_images(self, phase):
        Trainer.fixed_hr_images = Trainer.fixed_hr_images.to(self.config.training.device)
        Trainer.fixed_lr_images = Trainer.fixed_lr_images.to(self.config.training.device)
        upsampled_images = torch.nn.functional.interpolate(Trainer.fixed_lr_images.cpu(), scale_factor=4, mode="bicubic", antialias=True).to(self.config.training.device)
        self.writer.add_images(f"{phase}/HighRes", Trainer.fixed_hr_images, global_step=0)
        self.writer.add_images(f"{phase}/Bicubic", upsampled_images, global_step=0)

    @classmethod
    def _pre_train_setup(cls, dataloader):
        if cls.fixed_lr_images is None: 
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
        for step, (lr_images, hr_images) in tqdm(enumerate(train_dataloader, start=1), desc="Pretraining", total=len(train_dataloader)):
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
            if step % self.config.training.pretrain_log_iter == 0:
                self.generator.eval()
                with torch.no_grad():
                    fake_hr_images = (1.0 + self.generator(2.0*self.fixed_lr_images - 1.0)) / 2.0
                self.writer.add_images(
                    "Pretrain/Generated",
                    fake_hr_images,
                    global_step=step,
                )
                self.generator.train()
        torch.save({"model": self.generator.state_dict(), "optimizer": self.optim_generator.state_dict()}, f"runs/pretrain.pt")

    def train(self, train_dataloader, val_dataloader):
        self._calculate_metrics_over_dataset(val_dataloader, "GAN", step=0)
        if Trainer.fixed_lr_images is None:
            self._pre_train_setup(train_dataloader)
            self._log_fixed_images("GAN")
        real_labels = torch.ones((self.config.training.batch_size, 1), device=self.config.training.device)
        fake_labels = torch.zeros((self.config.training.batch_size, 1), device=self.config.training.device)
        self.generator.train()
        self.discriminator.train()
        for step, (lr_images, hr_images) in tqdm(enumerate(train_dataloader, start=1), desc="GAN Training", total=len(train_dataloader)):
            lr_images, hr_images = lr_images.to(
                self.config.training.device, non_blocking=True
            ), hr_images.to(self.config.training.device, non_blocking=True)
            self.optim_discriminator.zero_grad()
            y_real = self.discriminator(hr_images)
            sr_images = self.generator(lr_images).detach()
            y_fake = self.discriminator(sr_images)
            loss_real = self.loss_fn(y_real, real_labels)
            loss_fake = self.loss_fn(y_fake, fake_labels)
            discriminator_loss = 0.5 * loss_real + 0.5 * loss_fake
            discriminator_loss.backward()
            self.optim_discriminator.step()
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
            # Get the adv loss for the generator
            self.optim_generator.zero_grad()
            sr_images = self.generator(lr_images)
            y_fake = self.discriminator(sr_images)
            adv_loss = 0.5 * self.loss_fn(y_fake, real_labels)
            self.writer.add_scalar(
                "Loss/Generator/Adversarial",
                adv_loss,
                global_step=step,
            )
            # Get the content loss for the generator
            fake_features = self.perceptual_network(sr_images)
            real_features = self.perceptual_network(hr_images)
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
                    self._calculate_metrics_over_dataset(val_dataloader, "GAN", step=step)
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
