import torch
import segmentation_models_pytorch as smp


class UnetFeatureExtractor(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = smp.Unet(config.backbone_name, encoder_weights=config.encoder_weights, decoder_channels=config.decoder_channels)
        self.logits = torch.nn.Conv2d(config.decoder_channels[-1], 3, kernel_size=1, stride=1, bias=False)


    def forward(self, x):
        features = self.net.decoder(*self.net.encoder(x))
        logits = torch.tanh(self.logits(features))
        return features, logits


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


class UnetDiscriminator(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = smp.Unet(config.backbone_name, encoder_weights=config.encoder_weights, classes=3)

    def forward(self, x):
        return self.net(x)

