from torch import nn
from network_parts import DownSampling, UpSampling


class UnetGenerator2D(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.ds1 = DownSampling(in_channels=1, out_channels=filters, kernel_size=3, stride=1, padding=1)
        self.ds2 = DownSampling(in_channels=filters, out_channels=filters*2, kernel_size=3, stride=1, padding=1)
        self.ds3 = DownSampling(in_channels=filters*2, out_channels=filters*4, kernel_size=3, stride=1, padding=1)
        self.ds4 = DownSampling(in_channels=filters*4, out_channels=filters*8, kernel_size=3, stride=1, padding=1)
        self.ds5 = DownSampling(in_channels=filters*8, out_channels=filters*16, kernel_size=3, stride=1, padding=1)
        self.ds6 = DownSampling(in_channels=filters*16, out_channels=filters*32, kernel_size=3, stride=1, padding=1)
        self.up1 = UpSampling(in_channels=filters*32, out_channels=filters*16, kernel_size=3, stride=1, padding=1)
        self.up2 = UpSampling(in_channels=filters*2*16, out_channels=filters*8, kernel_size=3, stride=1, padding=1)
        self.up3 = UpSampling(in_channels=filters*2*8, out_channels=filters*4, kernel_size=3, stride=1, padding=1)
        self.up4 = UpSampling(in_channels=filters*2*4, out_channels=filters*2, kernel_size=3, stride=1, padding=1)
        self.up5 = UpSampling(in_channels=filters*2*2, out_channels=filters, kernel_size=3, stride=1, padding=1)
        self.up6 = UpSampling(in_channels=filters*2, out_channels=filters, kernel_size=3, stride=1, padding=1)
        self.up7 = UpSampling(in_channels=filters, out_channels=1, kernel_size=3, stride=1, padding=1, outermost=True)

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
        up7 = self.up7(up6, None)

        return up7



