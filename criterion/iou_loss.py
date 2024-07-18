"""
@author supermantx
@date 2024/7/12 11:40
"""
import torch
import math


class IOULoss:

    def __init__(self, box_format="xyxy", iou_type="iou", reduction="none", eps=1e-9):
        self.box_format = box_format
        self.iou_type = iou_type
        self.reduction = reduction
        self.eps = eps

    def __call__(self, boxes1, boxes2):
        if self.box_format.lower() == "xyxy":
            b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(boxes1, 1, dim=-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(boxes2, 1, dim=-1)

        elif self.box_format.lower() == "cxcywh":
            b1_cx, b1_cy, b1_w, b1_h = torch.split(boxes1, 1, dim=-1)
            b2_cx, b2_cy, b2_w, b2_h = torch.split(boxes2, 1, dim=-1)

            b1_x1, b1_y1, b1_x2, b1_x2 = \
                b1_cx - b1_w / 2, b1_cy - b1_h / 2, b1_cx + b1_w / 2, b2_cy + b1_h / 2
            b2_x1, b2_y1, b2_x2, b2_y2 = \
                b2_cx - b2_w / 2, b2_cy - b2_h / 2, b2_cx + b2_w / 2, b2_cy + b2_h / 2
        elif self.box_format.lower() == "xywh":
            b1_x1, b1_y1, b1_w, b1_h = torch.split(boxes1, 1, dim=-1)
            b2_x1, b2_y1, b2_w, b2_h = torch.split(boxes2, 1, dim=-1)
            b1_x2, b1_y2 = b1_x1 + b1_w, b1_y1 + b1_h
            b2_x2, b2_y2 = b2_x1 + b2_w, b2_y1 + b2_h
        inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                     (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union_area = w1 * h1 + w2 * h2 - inter_area
        iou = inter_area / (union_area + self.eps)
        outer_h = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        outer_w = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        if self.iou_type == "giou":
            outer_area = outer_w * outer_h
            iou = iou - (outer_area - union_area) / (outer_area + self.eps)
        elif self.iou_type == "siou":
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) / 2
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) / 2
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / outer_w) ** 2
            rho_y = (s_ch / outer_h) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss
