"""
@author supermantx
@date 2024/7/12 16:42
"""
from typing_extensions import ParamSpec, Self, TypeAlias
from typing import (
    Any,
    Dict,
    Iterable,
    Union,
)

import torch
from torch import nn
from torch.nn import Parameter
from yolo_duplicate.model import EfficientRep, DecoupleHead, RepBiPan

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class YOLOV6(nn.Module):

    def __init__(self):
        super(YOLOV6, self).__init__()
        self.backbone = EfficientRep()
        self.neck = RepBiPan()
        self.head = DecoupleHead()

    def get_learnable_parameter(self) -> ParamsT:
        res = []
        res.extend([v for k, v in self.backbone.named_parameters()])
        res.extend([v for k, v in self.neck.named_parameters()])
        res.extend([v for k, v in self.head.named_parameters()])
        return res

    def to(self, *args: Any, **kwargs: Any) -> Self:
        self.backbone.to(*args, **kwargs)
        self.neck.to(*args, **kwargs)
        self.head.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, X):
        c2, c3, c4, c5 = self.backbone(X)
        n3, n4, n5 = self.neck(c2, c3, c4, c5)
        out = self.head(n3, n4, n5)
        return out
