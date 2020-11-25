from torch import nn
from network_parts import Conv


class BaseDiscriminator(nn.Module):
    def __init__(self, filters, layer_count):
        super().__init__()
        layers = []
        layers += [Conv(in_channels=1, out_channels=filters, kernel_size=4, stride=2, padding=1, norm=False, activ="leaky")]
        in_channels = filters
        out_channels = in_channels * 2
        for i in range(layer_count-2):
            layers += [Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, activ="leaky")]
            in_channels = out_channels
            out_channels = out_channels*2

        layers += [Conv(in_channels=in_channels, out_channels=1, kernel_size=4, stride=2, padding=1, norm=False, activ=None)]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x