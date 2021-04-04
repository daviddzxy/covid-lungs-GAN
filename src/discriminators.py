from torch import nn
from network_parts import Conv


class PatchGanDiscriminator(nn.Module):
    def __init__(self, filters, depth, norm, in_channels=1):
        super().__init__()
        if depth < 2:
            raise ValueError

        if filters < 1:
            raise ValueError

        if norm == "batch_norm":
            norm = nn.BatchNorm2d
        elif norm == "instance_norm":
            norm = nn.InstanceNorm2d
        elif norm == "none":
            norm = None

        layers = []
        layers += [Conv(in_channels=in_channels, out_channels=filters, kernel_size=4, stride=2, padding=1, norm=None,
                        activ=nn.LeakyReLU(0.2, inplace=True))]
        in_channels = filters
        out_channels = in_channels * 2
        for i in range(depth-2):
            layers += [Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
                            padding=1, norm=norm, activ=nn.LeakyReLU(0.2, inplace=True))]
            in_channels = out_channels
            out_channels = out_channels * 2

        layers += [
                    Conv(in_channels=in_channels, out_channels=1, kernel_size=4, stride=2, padding=1, norm=None,
                         activ=None)
                  ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x