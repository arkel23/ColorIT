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
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class SELayer(nn.Module):
    def __init__(self, excite_dim, r_ratio=2, conv=False, squeeze_dim=None):
        """
        :param excite_dim: No of input channels
        :param r_ratio: By how much should the excite_dim should be reduced
        """
        super(SELayer, self).__init__()
        excite_dim_reduced = excite_dim // r_ratio
        self.r_ratio = r_ratio
        self.fc1 = nn.Linear(excite_dim, excite_dim_reduced, bias=True)
        self.fc2 = nn.Linear(excite_dim_reduced, excite_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if conv:
            self.conv = nn.Sequential(
                Rearrange('b s d -> b d s'),
                nn.Conv1d(squeeze_dim, 1, 1),
                Rearrange('b 1 s -> b s')
            )

    def forward(self, context, input=None):
        """
        :param context: X, shape = (batch_size, sequence_len, channels)
        :return: output tensor
        """
        # Average along all channels (channel squeeze)
        if hasattr(self, 'conv'):
            squeeze_tensor = self.conv(context)
        else:
            squeeze_tensor = reduce(context, 'b s d -> b s', 'mean')

        # excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        # reweight original feature maps based on computed excitation
        if input is not None:
            assert input.shape == context.shape, \
                'Cannot use channel squeeze and spatial excitation unless input/context are same size'
            output = torch.mul(input, rearrange(fc_out_2, 'b s -> b s 1'))
        else:
            output = torch.mul(context, rearrange(fc_out_2, 'b s -> b s 1'))
        return output

