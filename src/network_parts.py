from torch import nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=None, activ=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm
        if self.norm:
            self.norm = norm(out_channels)
        self.activ = activ

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x) if self.activ else x
        x = self.norm(x) if self.norm else x
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=None, outermost=False):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size, stride, padding, norm=norm,
                          activ=nn.ReLU(inplace=True))
        self.conv2 = Conv(out_channels, out_channels, kernel_size, stride, padding, norm=norm,
                          activ=nn.ReLU(inplace=True)) if outermost is False else Conv(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            norm=None,
            activ=nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, norm=nn.BatchNorm2d):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, padding_mode="reflect")]
        if norm is not None:
            layers += [norm(in_channels)]
        layers += [nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, padding_mode="reflect")
                   ]
        if norm is not None:
            layers += [norm(in_channels)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)
