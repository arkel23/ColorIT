from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange


class Depatchifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.depatchifier == 'inter_upsample_conv':
            self.inter = True
            self.gh = config.gh
            kernel_size = config.fh - 1
            padding = kernel_size // 2
            depatchifier = nn.Sequential(
                nn.Upsample(size=(config.image_size, config.image_size), mode='nearest'),
                nn.Conv2d(
                    in_channels=config.hidden_size * config.num_hidden_layers,
                    out_channels=config.num_channels,
                    kernel_size=kernel_size, stride=1, padding=padding
                )
            )

        elif config.depatchifier == 'transconv_single':
            stride = (config.slide_step, config.slide_step) if config.slide_step else (config.fh, config.fw)
            depatchifier = nn.Sequential(
                Rearrange('b (gh gw) d -> b d gh gw', gh=config.gh),
                nn.ConvTranspose2d(
                    in_channels=config.hidden_size, out_channels=config.num_channels,
                    kernel_size=(config.fh, config.fw), stride=stride)
            )
        elif config.depatchifier == 'upsample_conv_single':
            kernel_size = config.fh - 1
            padding = kernel_size // 2
            depatchifier = nn.Sequential(
                Rearrange('b (gh gw) d -> b d gh gw', gh=config.gh),
                nn.Upsample(size=(config.image_size, config.image_size), mode='nearest'),
                nn.Conv2d(
                    in_channels=config.hidden_size, out_channels=config.num_channels,
                    kernel_size=kernel_size, stride=1, padding=padding
                )
            )

        elif config.depatchifier == 'transconv_mult':
            if config.fh <= 4:
                depatchifier = nn.Sequential(
                    Rearrange('b (gh gw) d -> b d gh gw', gh=config.gh),
                    nn.ConvTranspose2d(
                        in_channels=config.hidden_size, out_channels=config.num_channels,
                        kernel_size=(config.fh, config.fw), stride=stride
                    ),
                    nn.ConvTranspose2d(
                        in_channels=config.hidden_size, out_channels=config.num_channels,
                        kernel_size=(config.fh, config.fw), stride=stride,
                    ),
                    nn.ConvTranspose2d(
                        in_channels=config.hidden_size, out_channels=config.num_channels,
                        kernel_size=(config.fh, config.fw), stride=stride,
                    )
                )
            else:
                depatchifier = nn.Sequential(
                    Rearrange('b (gh gw) d -> b d gh gw', gh=config.gh),
                    nn.ConvTranspose2d(
                        in_channels=config.hidden_size, out_channels=config.num_channels,
                        kernel_size=(config.fh, config.fw), stride=stride
                    ),
                    nn.ConvTranspose2d(
                        in_channels=config.hidden_size, out_channels=config.num_channels,
                        kernel_size=(config.fh, config.fw), stride=stride,
                    ),
                    nn.ConvTranspose2d(
                        in_channels=config.hidden_size, out_channels=config.num_channels,
                        kernel_size=(config.fh, config.fw), stride=stride
                    ),
                    nn.ConvTranspose2d(
                        in_channels=config.hidden_size, out_channels=config.num_channels,
                        kernel_size=(config.fh, config.fw), stride=stride,
                    ),
                    nn.ConvTranspose2d(
                        in_channels=config.hidden_size, out_channels=config.num_channels,
                        kernel_size=(config.fh, config.fw), stride=stride,
                    )
                )
        self.depatchifier = depatchifier

    def forward(self, x):
        if hasattr(self, 'inter'):
            x = rearrange(x, 'l b (gh gw) d -> b (l d) gh gw', gh=self.gh)

        x = self.depatchifier(x)
        return x
