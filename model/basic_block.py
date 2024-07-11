"""
@author supermantx
@date 2024/7/2 11:37
"""
import torch
from torch import nn


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky':
        return nn.LeakyReLU(0.1, inplace=True)
    elif activation == 'silu':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Unknown activation: {activation}")


class ConvBnAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, activation="relu", bias=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(RepVGGBlock, self).__init__()
        self.blocks = []
        block1 = ConvBnAct(in_channels, out_channels, 3, stride, 1)
        self.blocks.append(block1)
        block2 = ConvBnAct(in_channels, out_channels, 1, stride, 0)
        self.blocks.append(block2)
        if in_channels == out_channels and stride == 1:
            block3 = nn.Identity()
            self.blocks.append(block3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if i == 0:
                y = block(x)
            else:
                y += block(x)
        return self.activation(y)


class RepStageBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_blocks, stride):
        super(RepStageBlock, self).__init__()
        self.blocks = []
        for i in range(num_blocks):
            block = RepVGGBlock(in_channels, out_channels, stride if i == 0 else 1)
            self.blocks.append(block)
            in_channels = out_channels
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)


class BiC(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BiC, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
                                            kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.down_sample = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                     stride=2, padding=1)
        self.out = nn.Conv2d(in_channels=out_channels * 3, out_channels=out_channels, kernel_size=1)

    def forward(self, p, c_i, c_j):
        y = torch.concat([self.up_sample(p), self.conv1(c_i), self.down_sample(self.conv2(c_j))], dim=1)

        return self.out(y)


class CSPSPPF(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        blk1 = []
        blk1.append(nn.Conv2d(in_channels, in_channels, 1))
        blk1.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
        blk1.append(nn.Conv2d(in_channels, in_channels, 1))
        self.blk1 = nn.Sequential(*blk1)
        blk2 = []
        blk2.append(nn.Conv2d(in_channels * 4, in_channels, 1))
        blk2.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
        self.blk2 = nn.Sequential(*blk2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x):
        y = self.blk1(x)
        y1 = nn.MaxPool2d(5, stride=1, padding=2)(y)
        y2 = nn.MaxPool2d(5, stride=1, padding=2)(y)
        y3 = nn.MaxPool2d(5, stride=1, padding=2)(y)
        y = torch.cat([y, y1, y2, y3], dim=1)
        y1 = self.blk2(y)
        y2 = self.conv1(x)
        y = torch.cat([y1, y2], dim=1)
        return self.conv2(y)
