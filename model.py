import torch
from torchvision.models.vgg import VGG19_Weights, vgg19


class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:34]
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], requires_grad=False).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], requires_grad=False).view(1, 3, 1, 1),
        )

    def forward(self, x):
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return self.vgg(x)


class UpSamplingBlock(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=config.n_filters,
            out_channels=config.n_filters * 4,
            kernel_size=3,
            padding=1,
        )
        self.phase_shift = torch.nn.PixelShuffle(upscale_factor=2)
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        return self.relu(self.phase_shift(self.conv(x)))


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.InstanceNorm2d(out_channels)
        self.relu1 = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class Generator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=config.n_filters, kernel_size=3, padding=1),
            torch.nn.PReLU(),
        )
        self.stem = torch.nn.Sequential(
            *[
                ResidualBlock(in_channels=config.n_filters, out_channels=config.n_filters)
                for _ in range(config.n_layers)
            ]
        )

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=config.n_filters,
                out_channels=config.n_filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.InstanceNorm2d(config.n_filters),
        )

        self.upsampling = torch.nn.Sequential(
            UpSamplingBlock(config),
            UpSamplingBlock(config),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=config.n_filters,
                out_channels=3,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        residual = self.neck(x)
        x = self.stem(residual)
        x = self.bottleneck(x) + residual
        x = self.upsampling(x)
        return self.head(x)


class SimpleBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn = torch.nn.InstanceNorm2d(out_channels)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Discriminator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=config.n_filters, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )

        layers = [
            SimpleBlock(
                in_channels=config.n_filters,
                out_channels=config.n_filters,
                stride=2,
            ),
            SimpleBlock(
                in_channels=config.n_filters,
                out_channels=config.n_filters * 2,
                stride=1,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 2,
                out_channels=config.n_filters * 2,
                stride=2,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 2,
                out_channels=config.n_filters * 4,
                stride=1,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 4,
                out_channels=config.n_filters * 4,
                stride=2,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 4,
                out_channels=config.n_filters * 8,
                stride=1,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 8,
                out_channels=config.n_filters * 8,
                stride=2,
            ),
            torch.nn.Conv2d(
                in_channels=config.n_filters * 8, out_channels=1, kernel_size=1, padding=0, stride=1
            ),
        ]

        self.stem = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.neck(x)
        return self.stem(x)
