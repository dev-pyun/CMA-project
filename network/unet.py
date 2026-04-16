"""
UNet architecture for semantic segmentation.

Directly adapted from the deep-fmask project (Nambiar et al., 2022).
Supports variable depth and starting filter count for the multi-stage
self-training pipeline.

Architecture:
    Encoder: DoubleConv (Conv3x3-BN-ReLU x2 + Dropout) → MaxPool
    Decoder: ConvTranspose2d → Concat(skip) → DoubleConv
    Final:   Conv 1x1 → num_classes
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """
    Double convolution block:  Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → Dropout
    """

    def __init__(self, in_channels, out_channels, dropout=True):
        super(DoubleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if dropout:
            self.dropout = nn.Dropout2d(0.25)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """
    Encoder block: DoubleConv → MaxPool.
    Returns both the conv output (for skip connections) and the pooled output.
    """

    def __init__(self, in_channels, out_channels, pooling, dropout=True):
        super(Encoder, self).__init__()

        self.double_conv = DoubleConv(in_channels, out_channels, dropout)

        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.double_conv(x)
        double_conv_out = x
        x = self.pool(x)
        return x, double_conv_out


class Decoder(nn.Module):
    """
    Decoder block: Up-convolution → Concatenate with skip → DoubleConv.
    """

    def __init__(self, in_channels, out_channels, dropout=True):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.ConvTranspose2d(self.in_channels,
                                         self.out_channels,
                                         kernel_size=2,
                                         stride=2)

        self.double_conv = DoubleConv(in_channels=out_channels * 2,
                                      out_channels=out_channels,
                                      dropout=dropout)

    def forward(self, from_down, from_up):
        """
        Parameters
        ----------
        from_down : tensor from the encoder pathway (skip connection)
        from_up   : tensor from the previous decoder level
        """
        from_up = self.upconv(from_up)
        diff = from_down.size()[3] - from_up.size()[3]
        if diff:
            half_pad = diff // 2
            from_up = F.pad(from_up,
                            [half_pad, diff - half_pad,
                             half_pad, diff - half_pad])

        x = torch.cat((from_up, from_down), 1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    """
    UNet with configurable depth and filter count.

    Parameters
    ----------
    num_classes : int
        Number of output classes (6 for this project).
    in_channels : int
        Number of input channels (depends on band selection mode).
    depth : int
        Number of encoder levels (5 or 6 in the self-training pipeline).
    start_filts : int
        Number of filters in the first level (16, 24, or 32).
    dropout : bool
        Whether to use Dropout2d(0.25) in convolution blocks.
    """

    def __init__(self, num_classes=6, in_channels=7, depth=6,
                 start_filts=32, dropout=True):
        super(UNet, self).__init__()

        logger.info(f'UNet: in_channels={in_channels}, depth={depth}, '
                     f'start_filts={start_filts}, dropout={dropout}')

        # Build encoder
        self.encoders = []
        for i in range(depth):
            module_in_channels = in_channels if i == 0 else module_out_channels
            module_out_channels = start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = Encoder(module_in_channels, module_out_channels,
                                pooling=pooling, dropout=dropout)
            self.encoders.append(down_conv)
        self.encoders = nn.ModuleList(self.encoders)

        # Build decoder
        self.decoders = []
        for i in range(depth - 1):
            module_in_channels = module_out_channels
            module_out_channels = module_in_channels // 2
            up_conv = Decoder(module_in_channels, module_out_channels,
                              dropout=dropout)
            self.decoders.append(up_conv)
        self.decoders = nn.ModuleList(self.decoders)

        # Final 1×1 convolution
        self.conv_final = nn.Conv2d(module_out_channels,
                                    num_classes,
                                    kernel_size=1,
                                    stride=1)

        self.initialize_cnn_weights()

    def initialize_cnn_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        encoder_double_conv_outs = []

        for i, encoder in enumerate(self.encoders):
            x, double_conv_out = encoder(x)
            encoder_double_conv_outs.append(double_conv_out)

        # Reverse skip connections (exclude lowest level)
        encoder_double_conv_outs = encoder_double_conv_outs[-2::-1]
        for decoder, double_conv_out in zip(self.decoders,
                                            encoder_double_conv_outs):
            x = decoder(double_conv_out, x)

        x = self.conv_final(x)
        return x
