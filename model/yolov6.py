"""
@author supermantx
@date 2024/7/12 16:42
"""
import torch
from torch import nn

from yolo_duplicate.model import EfficientRep, DecoupleHead, RepBiPan


class YOLOV6(nn.Module):

    def __init__(self):
        super(YOLOV6, self).__init__()
        self.backbone = EfficientRep()
        self.neck = RepBiPan()
        self.head = DecoupleHead()

    def forward(self, X):
        c2, c3, c4, c5 = self.backbone(X)
        n3, n4, n5 = self.neck(c2, c3, c4, c5)
        out = self.head(n3, n4, n5)
        return out
