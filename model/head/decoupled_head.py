"""
@author supermantx
@date 2024/7/3 15:22
解耦预测头
"""
import torch
import torch.nn as nn

from yolo_duplicate.model.basic_block import ConvBnAct
from yolo_duplicate.model.get_build_config import *


class DecoupleHead(nn.Module):

    def __init__(self, num_class=80, ):
        super(DecoupleHead, self).__init__()
        build_lst = get_decouple_head_build_lst()
        self.stem1 = ConvBnAct(build_lst[0], build_lst[0],
                               kernel_size=1, stride=1,
                               padding=0, activation="silu")
        self.stem2 = ConvBnAct(build_lst[1], build_lst[1],
                               kernel_size=1, stride=1,
                               padding=0, activation="silu")
        self.stem3 = ConvBnAct(build_lst[2], build_lst[2],
                               kernel_size=1, stride=1,
                               padding=0, activation="silu")

        blk1_box = []
        blk1_cls = []
        blk2_box = []
        blk2_cls = []
        blk3_box = []
        blk3_cls = []

        blk1_box.append(ConvBnAct(build_lst[0], build_lst[0],
                                  kernel_size=3, stride=1,
                                  padding=1, activation="silu"))
        blk1_box.append(nn.Conv2d(build_lst[0], 4, 1))
        self.blk1_box = nn.Sequential(*blk1_box)
        blk1_cls.append(ConvBnAct(build_lst[0], build_lst[0],
                                  kernel_size=3, stride=1,
                                  padding=1, activation="silu"))
        blk1_cls.append(nn.Conv2d(build_lst[0], num_class, 1))
        self.blk1_cls = nn.Sequential(*blk1_cls)
        blk2_box.append(ConvBnAct(build_lst[1], build_lst[1],
                                  kernel_size=3, stride=1,
                                  padding=1, activation="silu"))
        blk2_box.append(nn.Conv2d(build_lst[1], 4, 1))
        self.blk2_box = nn.Sequential(*blk2_box)
        blk2_cls.append(ConvBnAct(build_lst[1], build_lst[1],
                                  kernel_size=3, stride=1,
                                  padding=1, activation="silu"))
        blk2_cls.append(nn.Conv2d(build_lst[1], num_class, 1))
        self.blk2_cls = nn.Sequential(*blk2_cls)
        blk3_box.append(ConvBnAct(build_lst[2], build_lst[2],
                                  kernel_size=3, stride=1,
                                  padding=1, activation="silu"))
        blk3_box.append(nn.Conv2d(build_lst[2], 4, 1))
        self.blk3_box = nn.Sequential(*blk3_box)
        blk3_cls.append(ConvBnAct(build_lst[2], build_lst[2],
                                  kernel_size=3, stride=1,
                                  padding=1, activation="silu"))
        blk3_cls.append(nn.Conv2d(build_lst[2], num_class, 1))
        self.blk3_cls = nn.Sequential(*blk3_cls)
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

    def forward(self, p5, p4, p3):
        p3 = self.stem1(p3)
        p4 = self.stem2(p4)
        p5 = self.stem3(p5)

        box1 = self.blk1_box(p3)
        box1 = box1.flatten(2).permute(0, 2, 1)
        cls1 = self.blk1_cls(p3)
        cls1 = torch.sigmoid(cls1)
        cls1 = cls1.flatten(2).permute(0, 2, 1)
        box2 = self.blk2_box(p4)
        box2 = box2.flatten(2).permute(0, 2, 1)
        cls2 = self.blk2_cls(p4)
        cls2 = torch.sigmoid(cls2)
        cls2 = cls2.flatten(2).permute(0, 2, 1)
        box3 = self.blk3_box(p5)
        box3 = box3.flatten(2).permute(0, 2, 1)
        cls3 = self.blk3_cls(p5)
        cls3 = torch.sigmoid(cls3)
        cls3 = cls3.flatten(2).permute(0, 2, 1)

        return (torch.cat([cls1, cls2, cls3], dim=1),
                torch.cat([box1, box2, box3], dim=1))


if __name__ == "__main__":
    model = DecoupleHead()
    input_size = [[1, 32, 80, 80], [1, 64, 40, 40], [1, 128, 20, 20]]
    input = []
    for size in input_size:
        torch.randn(size)
        input.append(torch.randn(size))
    y = model(*input)
    print(y[1].shape)
    print(y[2].shape)
