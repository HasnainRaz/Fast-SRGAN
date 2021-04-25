import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenet import InvertedResidual
from torchvision.models.vgg import vgg19
import torch


class Generator(nn.Module):
    def __init__(self, num_filters, num_blocks):
        super(Generator, self).__init__()
        self.base_conv = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.trunk = nn.Sequential(*[
            InvertedResidual(num_filters, num_filters, stride=1, expand_ratio=1)
            for _ in range(num_blocks)
        ])

        self.upsample = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 4,
                      kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_filters, num_filters * 4,
                      kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.final_conv = nn.Conv2d(
            num_filters, 3, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        features = self.base_conv(x)
        features = self.trunk(features)
        features = self.upsample(features)
        features = self.final_conv(features)
        upscaled_input = nn.functional.interpolate(
            x, scale_factor=4, mode='bilinear', align_corners=False)
        output = self.activation(features + upscaled_input)
        return output


class Discriminator(nn.Module):
    def __init__(self, num_filters, num_blocks=3):
        super(Discriminator, self).__init__()
        layer_sequence = [
            nn.Conv2d(3, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        num_filters_multiplier = 1
        num_filters_multiplier_previous = 1
        for n in range(1, num_blocks):
            num_filters_multiplier_previous = num_filters_multiplier
            num_filters_multiplier = min(2 ** n, 8)
            layer_sequence += [
                nn.Conv2d(
                    num_filters * num_filters_multiplier_previous,
                    num_filters * num_filters_multiplier, kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(num_filters * num_filters_multiplier),
                nn.LeakyReLU(0.2, True)
            ]

        num_filters_multiplier_previous = num_filters_multiplier
        num_filters_multiplier = min(2**num_blocks, 8)
        layer_sequence += [
            nn.Conv2d(
                num_filters * num_filters_multiplier_previous,
                num_filters * num_filters_multiplier,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_filters * num_filters_multiplier),
            nn.LeakyReLU(0.2, True)
        ]
        layer_sequence += [nn.Conv2d(num_filters * num_filters_multiplier,
                                     1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layer_sequence)

    def forward(self, x):
        return self.model(x)


class FeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34):
        super(FeatureExtractor, self).__init__()
        model = vgg19(pretrained=True)
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [-1, 1]
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return self.features(x)
