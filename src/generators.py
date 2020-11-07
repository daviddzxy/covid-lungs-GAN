from torch import nn, cat


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, activ='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        else:
            self.activ = nn.Tanh()
        self.norm = nn.BatchNorm2d(out_channels) if norm else None

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        x = self.norm(x) if self.norm else x
        return x


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            Conv(in_channels, out_channels),
            Conv(out_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, outermost=False):
        super().__init__()
        self.block = nn.Sequential(
            Conv(in_channels, out_channels),
            Conv(out_channels, out_channels) if outermost is False else Conv(
                                                                            out_channels,
                                                                            out_channels,
                                                                            norm=False,
                                                                            activ='tanh')
        )
        if outermost is False:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = None

    def forward(self, x1, x2):
        if x2 is not None:
            assert x1.shape[1] == x2.shape[1]
            x1 = cat((x1, x2), dim=1)

        x = self.block(x1)
        x = self.up(x) if self.up is not None else x
        return x


class UnetGenerator2D(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.ds1 = DownSampling(in_channels=1, out_channels=filters)
        self.ds2 = DownSampling(in_channels=filters, out_channels=filters*2)
        self.ds3 = DownSampling(in_channels=filters*2, out_channels=filters*4)
        self.ds4 = DownSampling(in_channels=filters*4, out_channels=filters*8)
        self.ds5 = DownSampling(in_channels=filters*8, out_channels=filters*16)
        self.ds6 = DownSampling(in_channels=filters*16, out_channels=filters*32)
        self.up1 = UpSampling(in_channels=filters*32, out_channels=filters*16)
        self.up2 = UpSampling(in_channels=filters*2*16, out_channels=filters*8)
        self.up3 = UpSampling(in_channels=filters*2*8, out_channels=filters*4)
        self.up4 = UpSampling(in_channels=filters*2*4, out_channels=filters*2)
        self.up5 = UpSampling(in_channels=filters*2*2, out_channels=filters)
        self.up6 = UpSampling(in_channels=filters*2, out_channels=1, outermost=True)

    def forward(self, x):
        down1 = self.ds1(x)
        down2 = self.ds2(down1)
        down3 = self.ds3(down2)
        down4 = self.ds4(down3)
        down5 = self.ds5(down4)
        down6 = self.ds6(down5)
        up1 = self.up1(down6, None)
        up2 = self.up2(up1, down5)
        up3 = self.up3(up2, down4)
        up4 = self.up4(up3, down3)
        up5 = self.up5(up4, down2)
        up6 = self.up6(up5, down1)
        return up6



