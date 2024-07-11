"""
@author supermantx
@date 2024/7/2 11:35
yolov6骨干网络
"""
import math

import torch
import torch.nn as nn

from yolo.model.basic_block import RepVGGBlock, RepStageBlock, ConvBnAct
from yolo.model.get_build_config import *


class EfficientRep(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()
        self.stem_layer = ConvBnAct(3, 16, 3, 2, 1)
        build_lst = get_efficientRep_build_lst()
        self.blocks = []
        for in_channels, out_channels, num_blocks in build_lst:
            self.blocks.append(RepStageBlock(in_channels, out_channels, num_blocks, 2))

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
