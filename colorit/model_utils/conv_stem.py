import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels, cifar=False, resnet=False,
                 pretrained=False, custom=None):
        super(ConvStem, self).__init__()
        if custom:
            self.resnet = timm.create_model(custom, pretrained=pretrained, num_classes=0, global_pool='')
        elif resnet:
            if out_channels == 192:
                stem = RN18(cifar=cifar, in_channels=in_channels, out_channels=out_channels)
            elif out_channels == 384:
                stem = RN34(cifar=cifar, in_channels=in_channels, out_channels=out_channels)
            elif out_channels == 768:
                stem = RN50(cifar=cifar, in_channels=in_channels, out_channels=out_channels)
        else:
            stem = ConvPatchingStem(in_channels, out_channels)

        if custom:
            out_channels_og = self.get_channels(self.resnet)
            self.conv1x1 = nn.Conv2d(in_channels=out_channels_og, out_channels=out_channels,
                                     kernel_size=1, stride=1)
        else:
            self.stem = stem

    def get_channels(self, stem):
        with torch.no_grad():
            x = torch.rand(2, 3, 224, 224)
            out = stem(x)
        return out.shape[1]

    def forward(self, x):
        if hasattr(self, 'conv1x1'):
            x = self.conv1x1(self.resnet(x))
        else:
            x = self.stem(x)        
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1,
                 activation=None, norm=None):
        super(ConvLayer, self).__init__()

        if in_channels == out_channels:
            stride = 1

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding))
        if activation == 'relu':
            self.activation = nn.ReLU()
        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x


class ConvPatchingStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPatchingStem, self).__init__()

        if out_channels == 192:
            channels_in_list = [in_channels, 16, 32, 64]
            channels_out_list = [16, 32, 64, 128]
        elif out_channels == 384:
            channels_in_list = [in_channels, 32, 64, 128]
            channels_out_list = [32, 64, 128, 256]
        elif out_channels == 768:
            channels_in_list = [in_channels, 64, 128, 128, 256, 256]
            channels_out_list = [64, 128, 128, 256, 256, 512]

        self.conv3x3layers = nn.ModuleList([
            ConvLayer(channels_in, channels_out)
            for (channels_in, channels_out) in zip(channels_in_list, channels_out_list)
        ])

        self.conv1x1 = nn.Conv2d(
            channels_out_list[-1], out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        for layer in self.conv3x3layers:
            x = layer(x)
        x = self.conv1x1(x)
        return x

