import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange

from .transformer import Transformer
from .squeeze_and_excitation import SEBlock


class Depatchifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        t_conv_kernel_size = config.fh
        t_conv_stride = config.slide_step if config.slide_step else config.fh

        conv_kernel_size = config.fh - 1
        conv_padding = conv_kernel_size // 2

        if config.depatchifier in ('transconv', 'transconv_ucatconv', 'transconv_se'):
            self.depatchifier = nn.Sequential(
                Rearrange('b (gh gw) d -> b d gh gw', gh=config.gh),
                nn.ConvTranspose2d(
                    in_channels=config.hidden_size, out_channels=config.num_channels,
                    kernel_size=t_conv_kernel_size, stride=t_conv_stride)
            )

            if config.depatchifier in ('transconv_ucatconv', 'transconv_se'):
                self.norm = nn.Sequential(
                    Rearrange('b c h w -> b h w c'),
                    nn.LayerNorm(config.num_channels, eps=config.layer_norm_eps),
                    nn.GELU(),
                    Rearrange('b c h w')
                )

            if config.depatchifier == 'transconv_ucatconv':
                self.ucat_conv = nn.Conv2d(
                    in_channels=config.num_channels * 2, out_channels=config.num_channels,
                    kernel_size=conv_kernel_size, padding=conv_padding, stride=1)
            elif config.depatchifier == 'transconv_se':
                self.se = SEBlock(config.se, config.seq_len, config.num_channels, config.se_ratio,
                                  config.se_residual, config.se_reweight_target)

        elif config.depatchifier == 'upsample_conv':
            self.depatchifier = nn.Sequential(
                Rearrange('b (gh gw) d -> b d gh gw', gh=config.gh),
                nn.Upsample(size=(config.image_size, config.image_size), mode='nearest'),
                nn.Conv2d(
                    in_channels=config.hidden_size, out_channels=config.num_channels,
                    kernel_size=conv_kernel_size, padding=conv_padding, stride=1,
                )
            )

        elif config.depatchifier == 'inter_upsample_conv':
            self.inter = True
            self.gh = config.gh
            kernel_size = config.fh - 1
            padding = kernel_size // 2
            self.upsample = nn.Upsample(size=(config.image_size, config.image_size), mode='nearest')
            self.depatchifier = nn.Conv2d(
                in_channels=(config.hidden_size * config.num_hidden_layers) + 3,
                out_channels=config.num_channels,
                kernel_size=kernel_size, stride=1, padding=padding
            )

        elif config.depatchifier == 'inter_upsample_conv_conv':
            self.inter = True
            self.gh = config.gh
            kernel_size = config.fh - 1
            padding = kernel_size // 2

            self.conv_stage_1 = nn.ModuleList(
                Rearrange('b (h w) c -> b c h w', h=self.gh),
                nn.Upsample(size=(config.image_size, config.image_size), mode='nearest'),
                nn.Conv2d(in_channels=config.hidden_size, out_channels=config.num_channels,
                          kernel_size=kernel_size, stride=1, padding=padding),
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(config.num_channels, eps=config.layer_norm_eps),
                Rearrange('b h w c -> b c h w')
            )

            self.depatchifier = nn.Conv2d(
                in_channels=(config.num_channels * config.num_hidden_layers + 1),
                out_channels=config.num_channels,
                kernel_size=kernel_size, stride=1, padding=padding
            )

        elif config.depatchifier == 'upsample_convnext':
            self.context = True
            # Transformer encoder
            self.depatchifier = nn.Sequential(
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

        else:
            raise NotImplementedError

    def forward(self, x, images):
        if hasattr(self, 'inter'):
            if hasattr(self, 'conv_stage_1'):
                x = [self.conv_stage_1[i](ft) for i, ft in enumerate(x)]
                x.apppend(images)
                x = rearrange(x, 'l b c h w -> b (l d)')
            else:
                x = rearrange(x, 'l b (gh gw) d -> b (l d) gh gw', gh=self.gh)
                x = self.upsample(x)
                x = torch.cat([x, images], dim=1)

        if hasattr(self, 'context'):
            x = self.depatchifier(x, images)
        else:
            x = self.depatchifier(x)

        if hasattr(self, 'norm'):
            x = self.norm(x)

        if hasattr(self, 'ucat_conv'):
            x = torch.cat([x, images], dim=1)
            x = self.ucat_conv(x)
        elif hasattr(self, 'se'):
            x = self.se(images, x)

        return x
