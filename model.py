from functools import partial

import torch.nn as nn
from torchvision.models.vgg import vgg19


class ConvBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, activation='prelu'):
        padding = (kernel_size - 1) // 2
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.PReLU() if activation.lower() == 'prelu' else nn.LeakyReLU()
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBlock(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBlock(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


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


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        self.net = vgg19(pretrained=True)
        del self.net.avgpool
        del self.net.classifier

        self.net = nn.Sequential(*list(self.net.features.children())[:-2])

    def forward(self, x):
        return self.net(x)


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
        self.conv1 = ConvBlock(3, cfg.DISCRIMINATOR.FEATURES, norm_layer=nn.BatchNorm2d, activation='leaky')
        in_features = cfg.DISCRIMINATOR.FEATURES
        out_features = cfg.DISCRIMINATOR.FEATURES
        strides = 1
        layers = []
        for i in range(cfg.DISCRIMINATOR.NUM_BLOCKS):
            layers.append(
                ConvBlock(in_features,
                          out_features,
                          norm_layer=nn.BatchNorm2d,
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
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(in_features * 6 * 6, 1024)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.validity = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stem(x)
        x = self.adaptive_pool(x)
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.leaky_relu(x)
        return self.validity(x)
