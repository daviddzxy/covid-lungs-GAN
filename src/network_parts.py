from torch import nn, cat


class Conv(nn.Module):
    #TODO add Instance norm option
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=True, activ="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if activ == "relu":
            self.activ = nn.ReLU(inplace=True)
        elif activ == "tanh":
            self.activ = nn.Tanh()
        else:
            self.activ = None
        self.norm = nn.BatchNorm2d(out_channels) if norm else None

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x) if self.activ else x
        x = self.norm(x) if self.norm else x
        return x


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size, stride, padding),
            Conv(out_channels, out_channels, kernel_size, stride, padding),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, outermost=False):
        super().__init__()
        self.block = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size, stride, padding),
            Conv(out_channels, out_channels, kernel_size, stride, padding) if outermost is False else Conv(
                                                                            out_channels,
                                                                            out_channels,
                                                                            kernel_size,
                                                                            stride,
                                                                            padding,
                                                                            norm=False,
                                                                            activ="tanh")
        )
        if outermost is False:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = None

    def forward(self, x1, x2):
        if x2 is not None:
            assert x1.shape[1] == x2.shape[1]
            x1 = cat((x1, x2), dim=1)

        x = self.block(x1)
        x = self.up(x) if self.up is not None else x
        return x