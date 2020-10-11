import torch.nn as nn
from torchvision.models.mobilenet import InvertedResidual, ConvBNReLU, \
    MobileNetV2, model_urls, load_state_dict_from_url


class MobileNetEncoder(MobileNetV2):
    def __init__(self):
        super(MobileNetEncoder, self).__init__(num_classes=1000)
        self.load_state_dict(load_state_dict_from_url(model_urls['mobilenet_v2']))
        del self.classifier

    def _forward_impl(self, x):
        return self.features(x)


class FastGenerator(nn.Module):
    """SRGAN Generator for fast super resolution."""

    def __init__(self, cfg):
        """
        Initializes the Mobile SRGAN class.
        Args:
            cfg: Dict, Parameters used to construct and initialize the model.
        Returns:
            None
        """
        super(FastGenerator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, cfg.GENERATOR.FEATURES, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cfg.GENERATOR.FEATURES),
            nn.PReLU()
        )

        self.blocks = nn.Sequential(
            *[
                InvertedResidual(cfg.GENERATOR.FEATURES,
                                 cfg.GENERATOR.FEATURES,
                                 stride=1,
                                 expand_ratio=cfg.GENERATOR.EXPANSION_FACTOR)
                for _ in range(cfg.GENERATOR.NUM_BLOCKS)
            ]
        )

        self.upsampling = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(cfg.GENERATOR.FEATURES, cfg.GENERATOR.FEATURES, kernel_size=3, padding=1),
            nn.PReLU(),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(cfg.GENERATOR.FEATURES, cfg.GENERATOR.FEATURES, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.final_conv = nn.Conv2d(cfg.GENERATOR.FEATURES, 3, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.upsampling(x)
        x = self.final_conv(x)
        return self.activation(x)


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBNReLU(3, cfg.DISCRIMINATOR.FEATURES)
        self.stem = nn.Sequential(
            *[
                ConvBNReLU(cfg.DISCRIMINATOR.FEATURES * 2 if i % 2 != 0 else cfg.DISCRIMINATOR.FEATURES,
                           cfg.DISCRIMINATOR.FEATURES * 2 if i % 2 == 0 else cfg.DISCRIMINATOR.FEATURES,
                           stride=2 if i % 2 == 0 else 2,
                           kernel_size=3)
                for i in range(cfg.DISCRIMINATOR.NUM_BLOCKS)
            ]
        )

        self.validity = nn.Conv2d(cfg.DISCRIMINATOR.FEATURES
                                  if cfg.DISCRIMINATOR.NUM_BLOCKS % 2 == 0
                                  else cfg.DISCRIMINATOR.FEATURES * 2,
                                  1,
                                  kernel_size=1,
                                  padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stem(x)
        return self.validity(x)
