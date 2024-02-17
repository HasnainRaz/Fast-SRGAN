import torch
from torchvision.models.vgg import vgg19, VGG19_Weights
from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights, LayerNorm2d
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights

class VGG19(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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


class SwinT(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 7th Block is the deepest
        self.net = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).features[:8]
        for param in self.net.parameters():
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
        return self.net(x)


class ConvNeXt(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 7th is the deepest block
        self.net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features[:8]
        for param in self.net.parameters():
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
        return self.net(x)

class PerceptualNetwork(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.name == "vgg":
            self.net = VGG19(config)
        elif config.name == "swin":
            self.net = SwinT()
        elif config.name == "convnext":
            self.net = ConvNeXt()

    def forward(self, x):
        return self.net(x)


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
            bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.act1 = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + residual


class Generator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=config.n_filters, kernel_size=9, padding=4
            ),
            torch.nn.PReLU(),
        )
        self.stem = torch.nn.Sequential(
            *[
                ResidualBlock(
                    in_channels=config.n_filters, out_channels=config.n_filters
                )
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
            torch.nn.BatchNorm2d(config.n_filters),
        )

        self.upsampling = torch.nn.Sequential(
            UpSamplingBlock(config),
            UpSamplingBlock(config),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=config.n_filters,
                out_channels=3,
                kernel_size=9,
                padding=4,
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
            bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class SimpleDiscriminator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=config.n_filters, kernel_size=3, padding=1
            ),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )

        layers = [
            SimpleBlock(
                in_channels=config.n_filters,
                out_channels=config.n_filters,
                stride=2,
            ),
            SimpleBlock(
                in_channels=config.n_filters ,
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
        ]

        self.stem = torch.nn.Sequential(*layers)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(config.n_filters * 8 * 6 * 6, 1024, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 1, bias=False),
        )

    def forward(self, x):
        x = self.neck(x)
        x = self.stem(x).view(-1, self.config.n_filters * 8 * 6 * 6)
        return self.head(x)


class SwinDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.net.head = torch.nn.Linear(768, 1)

    def forward(self, x):
        return self.net(x)


class ConvNeXtDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.net.classifier = torch.nn.Sequential(
            LayerNorm2d(768),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(768, 1),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.name == "simple":
            self.net = SimpleDiscriminator(config)
        elif config.name == "swin":
            self.net = SwinDiscriminator()
        elif config.name == "convnext":
            self.net = ConvNeXtDiscriminator()

    def forward(self, x):
        return self.net(x)

