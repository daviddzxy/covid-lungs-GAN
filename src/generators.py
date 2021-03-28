from torch import nn, cat
from network_parts import DoubleConv, ResidualBlock


class UnetGenerator2D(nn.Module):
    def __init__(self, depth, filters):
        super().__init__()
        if filters < 1:
            raise ValueError

        if depth < 2:
            raise ValueError

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        in_channels = 1
        out_channels = filters
        for i in range(depth - 1):
            self.down_layers += [DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                            padding=1)]
            in_channels = out_channels
            out_channels = out_channels * 2

        out_channels = in_channels
        self.bottleneck = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                     padding=1)
        out_channels = out_channels // 2

        for i in range(depth - 2):
            self.up_layers += [DoubleConv(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=3,
                                          stride=1,
                                          padding=1)]
            in_channels = in_channels // 2
            out_channels = out_channels // 2

        self.outermost = DoubleConv(in_channels=in_channels * 2, out_channels=1, kernel_size=3, stride=1, padding=1,
                                    outermost=True)

    def forward(self, x):
        down_outputs = []
        x = self.down_layers[0](x)
        down_outputs += [x]
        for down_layer in self.down_layers[1:]:
            x = down_layer(self.maxpool(x))
            down_outputs += [x]

        x = self.upsample(self.bottleneck(self.maxpool(x)))

        for down_output, up_layer in zip(reversed(down_outputs), self.up_layers):
            x = self.upsample(up_layer(cat((x, down_output), dim=1)))

        x = self.outermost(cat((x, down_outputs[0]), dim=1))
        return x


class ResNetGenerator2D(nn.Module):
    def __init__(self, resnet_depth, scale_depth, filters):
        super().__init__()
        if resnet_depth < 1:
            raise ValueError

        if filters < 1:
            raise ValueError

        if scale_depth < 1:
            raise ValueError

        layers = []
        mult = 1
        #  downsampling path
        for i in range(0, scale_depth):
            layers += [
                nn.Conv2d(in_channels=mult, out_channels=mult*2, kernel_size=4, stride=2,
                          padding=1,
                          padding_mode="reflect"),
                nn.BatchNorm2d(mult*2),
                nn.ReLU(inplace=True)
            ]
            mult = mult*2

        #  residual path
        for i in range(0, resnet_depth):
            layers += [ResidualBlock(in_channels=mult, kernel_size=3)]

        #  upsampling path
        for i in range(0, scale_depth-1):
            layers += [
                nn.ConvTranspose2d(in_channels=mult,
                                   out_channels=mult // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1),
                nn.BatchNorm2d(mult // 2),
                nn.ReLU(inplace=True)
            ]
            mult = mult // 2

        layers += [
            nn.ConvTranspose2d(in_channels=mult,
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.Tanh()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
