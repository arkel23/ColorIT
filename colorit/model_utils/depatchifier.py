import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange

from .transformer import Transformer


class Depatchifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.depatchifier == 'inter_upsample_conv':
            self.inter = True
            self.gh = config.gh
            kernel_size = config.fh - 1
            padding = kernel_size // 2
            self.upsample = nn.Upsample(size=(config.image_size, config.image_size), mode='nearest')
            depatchifier = nn.Conv2d(
                in_channels=(config.hidden_size * config.num_hidden_layers) + 3,
                out_channels=config.num_channels,
                kernel_size=kernel_size, stride=1, padding=padding
            )

        elif config.depatchifier == 'upsample_convnext':
            self.context = True
            # Transformer encoder
            depatchifier = nn.Sequential(
                nn.Upsample(size=(config.image_size, config.image_size), mode='nearest'),
                Transformer(
                    num_layers=config.num_hidden_layers // 2,
                    dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    ff_dim=config.intermediate_size,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    layer_norm_eps=config.layer_norm_eps,
                    sd=config.sd,
                    attn='conv',
                    seq_len=config.seq_len,
                    ret_inter=False
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

    def forward(self, x, images):
        if hasattr(self, 'inter'):
            x = rearrange(x, 'l b (gh gw) d -> b (l d) gh gw', gh=self.gh)
            x = self.upsample(x)
            x = torch.cat([x, images], dim=1)

        if hasattr(self, 'context'):
            x = self.depatchifier(x, images)
        else:
            x = self.depatchifier(x)
        return x
