"""
@author supermantx
@date 2024/7/2 11:35
yolov6骨干网络
"""
import math

import torch
import torch.nn as nn

from yolo_duplicate.model.basic_block import RepVGGBlock, RepStageBlock, ConvBnAct
from yolo_duplicate.model.get_build_config import *


class EfficientRep(nn.Module):

    def __init__(self, cfg=None):
        super(EfficientRep, self).__init__()
        self.stem_layer = ConvBnAct(3, 16, 3, 2, 1)
        build_lst = get_efficientRep_build_lst()
        self.blocks = []
        for in_channels, out_channels, num_blocks in build_lst:
            self.blocks.append(RepStageBlock(in_channels, out_channels, num_blocks, 2))
        self.blocks = nn.Sequential(*self.blocks)
        self.apply(self._init_weight)

    @staticmethod
    def _init_weight(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        outputs = []
        x = self.stem_layer(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            outputs.append(x)
        return outputs


if __name__ == "__main__":
    model = EfficientRep()
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    for i in y:
        print(i.shape)
