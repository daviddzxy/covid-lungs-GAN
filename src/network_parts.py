from torch import nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=True, activ="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if activ == "relu":
            self.activ = nn.ReLU(inplace=True)
        elif activ == "tanh":
            self.activ = nn.Tanh()
        elif activ == "leaky":
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activ = None
        self.norm = nn.BatchNorm2d(out_channels) if norm else None

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x) if self.activ else x
        x = self.norm(x) if self.norm else x
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, outermost=False):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = Conv(out_channels, out_channels, kernel_size, stride, padding) if outermost is False else Conv(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            norm=False,
            activ="tanh")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, norm_layer=nn.BatchNorm2d):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, padding_mode="reflect"),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, padding_mode="reflect"),
            norm_layer(in_channels)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)
