from torch import nn, cat
from network_parts import DoubleConv


class UnetGenerator2D(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.down1 = DoubleConv(in_channels=1, out_channels=filters, kernel_size=3, stride=1, padding=1)
        self.down2 = DoubleConv(in_channels=filters, out_channels=filters*2, kernel_size=3, stride=1, padding=1)
        self.down3 = DoubleConv(in_channels=filters*2, out_channels=filters*4, kernel_size=3, stride=1, padding=1)
        self.down4 = DoubleConv(in_channels=filters*4, out_channels=filters*8, kernel_size=3, stride=1, padding=1)
        self.down5 = DoubleConv(in_channels=filters*8, out_channels=filters*16, kernel_size=3, stride=1, padding=1)
        self.conv_bottleneck = DoubleConv(in_channels=filters*16, out_channels=filters*16, kernel_size=3, stride=1, padding=1)
        self.up1 = DoubleConv(in_channels=filters*2*16, out_channels=filters*8, kernel_size=3, stride=1, padding=1)
        self.up2 = DoubleConv(in_channels=filters*2*8, out_channels=filters*4, kernel_size=3, stride=1, padding=1)
        self.up3 = DoubleConv(in_channels=filters*2*4, out_channels=filters*2, kernel_size=3, stride=1, padding=1)
        self.up4 = DoubleConv(in_channels=filters*2*2, out_channels=filters, kernel_size=3, stride=1, padding=1)
        self.up5 = DoubleConv(in_channels=filters*2, out_channels=1, kernel_size=3, stride=1, padding=1, outermost=True)


    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.maxpool(x1))
        x3 = self.down3(self.maxpool(x2))
        x4 = self.down4(self.maxpool(x3))
        x5 = self.down5(self.maxpool(x4))
        x6 = self.upsample(self.conv_bottleneck(self.maxpool(x5)))
        x = self.upsample(self.up1(cat((x6, x5), dim=1)))
        x = self.upsample(self.up2(cat((x, x4), dim=1)))
        x = self.upsample(self.up3(cat((x, x3), dim=1)))
        x = self.upsample(self.up4(cat((x, x2), dim=1)))
        x = self.up5(cat((x, x1), dim=1))
        return x



