"""
@author supermantx
@date 2024/7/2 15:16
脖子
"""
import math

import torch
import torch.nn as nn

from yolo_duplicate.model.basic_block import CSPSPPF, BiC, RepStageBlock, ConvBnAct
from yolo_duplicate.model.get_build_config import *

class RepBiPan(nn.Module):

    def __init__(self):
        super().__init__()
        num_repeats, build_lst = get_reoBiPan_build_lst()
        self.cspsppf = CSPSPPF(build_lst[0])
        self.conv1 = ConvBnAct(build_lst[0], build_lst[1], 1, padding=0)
        self.bic1 = BiC(in_channels=[build_lst[2], build_lst[1]], out_channels=build_lst[1])
        self.blk1 = RepStageBlock(build_lst[1], build_lst[1], num_repeats[0], 1)
        self.conv2 = ConvBnAct(build_lst[4], build_lst[3], 1, padding=0)
        self.bic2 = BiC(in_channels=[build_lst[4], build_lst[3]], out_channels=build_lst[3])
        self.blk2 = RepStageBlock(build_lst[3], build_lst[3], num_repeats[1], 1)

        self.conv3 = ConvBnAct(build_lst[3], build_lst[4], 3, stride=2, padding=1)
        self.blk3 = RepStageBlock(build_lst[4] * 2, build_lst[4], num_repeats[2], 1)
        self.conv4 = ConvBnAct(build_lst[4], build_lst[4], 3, stride=2, padding=1)
        self.blk4 = RepStageBlock(build_lst[5], build_lst[5], num_repeats[3], 1)

    def forward(self, c2, c3, c4, c5):
        p5 = self.cspsppf(c5)
        p5 = self.conv1(p5)
        p4 = self.bic1(p5, c4, c3)
        p4 = self.blk1(p4)
        p3 = self.conv2(p4)
        p3 = self.bic2(p3, c3, c2)
        p3 = self.blk2(p3)

        n3 = p3
        n4 = self.conv3(p3)
        n4 = self.blk3(torch.concat([n4, p4], dim=1))
        n5 = self.conv4(n4)
        n5 = self.blk4(torch.concat([n5, p5], dim=1))
        return n3, n4, n5



if __name__ == '__main__':
    input_size = [[1, 32, 160, 160], [1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]]
    model = RepBiPan()
    input = []
    for size in input_size:
        torch.randn(size)
        input.append(torch.randn(size))
    y = model(*input)
    for i in y:
        print(i.shape)