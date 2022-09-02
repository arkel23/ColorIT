"""
Squeeze and Excitation Module
*****************************
Collection of squeeze and excitation classes where each can be inserted as a block into a neural network architechture
    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
"""

import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class SEBlock(nn.Module):
    def __init__(self, se, spatial_dim, channel_dim, ratio=4, weighted=False, res=False, reweight_target=True):
        super(SEBlock, self).__init__()
        if 'w' in se:
            se = se.split('_')[0]
            weighted = True

        self.res = res
        self.reweight_target = reweight_target
        assert self.res or self.reweight_target, 'Either res or reweight_target must be True'

        if se == 'ssce':
            self.se = SELayer(
                excite_dim=channel_dim, squeeze_dim=spatial_dim, ratio=ratio, weighted=weighted)
        elif se == 'csse':
            self.se = SELayer(
                excite_dim=spatial_dim, squeeze_dim=channel_dim, ratio=ratio, weighted=weighted, spatial_squeeze=False)
        else:
            self.se = CombinedSELayer(spatial_dim, channel_dim, ratio=4, weighted=weighted)

    def forward(self, x, target):
        if self.res and self.reweight_target:
            h = self.se(x, target)
            x = target + h
        elif self.res:
            h = self.se(x)
            x = target + h
        elif self.reweight_target:
            x = self.se(x, target)
        else:
            raise NotImplementedError
        return x


class SELayer(nn.Module):
    def __init__(self, excite_dim, squeeze_dim, ratio=2,
                 spatial_squeeze=True, weighted=False):
        """
        :param excite_dim: No of input channels
        :param r_ratio: By how much should the excite_dim should be reduced
        """
        super(SELayer, self).__init__()
        excite_dim_reduced = int(excite_dim * ratio)
        self.fc1 = nn.Linear(excite_dim, excite_dim_reduced, bias=True)
        self.fc2 = nn.Linear(excite_dim_reduced, excite_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.spatial_squeeze = spatial_squeeze
        if self.spatial_squeeze and weighted:
            self.weighted_pool = nn.Sequential(
                Rearrange('b c h w -> b c (h w)'),
                nn.Linear(squeeze_dim, 1),
                Rearrange('b c 1 -> b c')
            )
        elif not self.spatial_squeeze and weighted:
            self.weighted_pool = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                nn.Linear(squeeze_dim, 1),
                Rearrange('b h w 1 -> b (h w)')
            )

    def forward(self, x, target=None):
        """
        :param x: X, shape = (batch_size, sequence_len, channels)
        :return: output tensor
        """
        b, c, h, w = x.shape

        if self.spatial_squeeze:
            # Average along all spatial positions (spatial squeeze)
            if hasattr(self, 'weighted_pool'):
                squeeze_tensor = self.weighted_pool(x)
            else:
                squeeze_tensor = reduce(x, 'b c h w -> b c', 'mean')
        else:
            # Average along all channels (channel squeeze)
            if hasattr(self, 'weighted_pool'):
                squeeze_tensor = self.weighted_pool(x)
            else:
                squeeze_tensor = reduce(x, 'b c h w -> b (h w)', 'mean')

        # excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        if self.spatial_squeeze:
            # reweight original channels based on computed excitation
            if target is not None:
                output = torch.mul(target, rearrange(fc_out_2, 'b c -> b c 1 1'))
            else:
                output = torch.mul(x, rearrange(fc_out_2, 'b c -> b c 1 1'))
        else:
            # reweight original feature maps based on computed excitation
            if target is not None:
                assert target.shape == x.shape, \
                    'Cannot use channel squeeze and spatial excitation unless target/x are same size'
                output = torch.mul(target, rearrange(fc_out_2, 'b (h w) -> b 1 h w', h=h))
            else:
                output = torch.mul(x, rearrange(fc_out_2, 'b (h w) -> b 1 h w', h=h))
        return output


class CombinedSELayer(nn.Module):
    def __init__(self, spatial_dim, channel_dim, ratio=4, weighted=False):
        """
        :param excite_dim: No of input channels
        :param r_ratio: By how much should the excite_dim should be reduced
        """
        super(CombinedSELayer, self).__init__()
        self.sSE = SELayer(
            excite_dim=channel_dim, squeeze_dim=spatial_dim, ratio=ratio, weighted=weighted)
        self.cSE = SELayer(
            excite_dim=spatial_dim, squeeze_dim=channel_dim, ratio=ratio, weighted=weighted, spatial_squeeze=False)

    def forward(self, x, target=None):
        """
        :param x: X, shape = (batch_size, seq_len, channels)
        :return: output
        """
        output = torch.max(self.sSE(x, target), self.cSE(x, target))
        return output
