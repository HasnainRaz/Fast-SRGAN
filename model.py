from functools import partial

import torch.nn as nn
from torchvision.models.mobilenet import InvertedResidual, ConvBNReLU, \
    MobileNetV2, model_urls as mobile_urls, load_state_dict_from_url
from torchvision.models.vgg import VGG, make_layers, cfgs, model_urls as vgg_urls


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class MobileNetEncoder(MobileNetV2):
    def __init__(self):
        super(MobileNetEncoder, self).__init__(num_classes=1000)
        self.load_state_dict(load_state_dict_from_url(mobile_urls['mobilenet_v2']))
        del self.classifier

    def _forward_impl(self, x):
        return self.features(x)


class VGGEncoder(VGG):
    def __init__(self):
        features = make_layers(cfgs['D'])
        super(VGGEncoder, self).__init__(features)
        self.load_state_dict(load_state_dict_from_url(vgg_urls['vgg16']))
        del self.classifier
        del self.avgpool
        self.features = nn.Sequential(*[x for x in list(self.features.children())[:-2]])

    def forward(self, x):
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
        if cfg.BLOCK == 'residual':
            block = ResidualBlock
        elif cfg.BLOCK == 'inverted_residual':
            block = partial(InvertedResidual, stride=1, expand_ratio=cfg.GENERATOR.EXPANSION_FACTOR)
        else:
            raise ValueError(
                "Only 'residual' or 'inverted_residual' blocks are supported, please specify one in the config ")
        self.blocks = nn.Sequential(
            *[
                block(cfg.GENERATOR.FEATURES, cfg.GENERATOR.FEATURES)
                for _ in range(cfg.GENERATOR.NUM_BLOCKS)
            ]
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(cfg.GENERATOR.FEATURES, cfg.GENERATOR.FEATURES * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),

            nn.Conv2d(cfg.GENERATOR.FEATURES, cfg.GENERATOR.FEATURES * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
        self.final_conv = nn.Conv2d(cfg.GENERATOR.FEATURES, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.upsampling(x)
        x = self.final_conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBNReLU(3, cfg.DISCRIMINATOR.FEATURES)
        in_features = cfg.DISCRIMINATOR.FEATURES
        out_features = cfg.DISCRIMINATOR.FEATURES
        strides = 1
        layers = []
        for i in range(cfg.DISCRIMINATOR.NUM_BLOCKS):
            layers.append(
                ConvBNReLU(in_features,
                           out_features,
                           stride=strides,
                           kernel_size=3)
            )
            if i % 2 != 0:
                out_features = in_features * 2
                strides = 2
            else:
                in_features = out_features
                strides = 1
        self.stem = nn.Sequential(*layers)
        self.validity = nn.Conv2d(in_features,
                                  1,
                                  kernel_size=1,
                                  padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stem(x)
        return self.validity(x)
