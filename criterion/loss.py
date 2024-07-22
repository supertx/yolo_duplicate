"""
@author supermantx
@date 2024/7/4 15:49
计算loss
"""

import torch
from torch import nn
from torch.nn import functional as F

from yolo_duplicate.assigners import ATSSAssigner, TaskAlignedAssigner, generate_anchor
from yolo_duplicate.criterion.iou_loss import IOULoss


class ComputeLoss(nn.Module):

    def __init__(self,
                 fpn_strides=None,
                 num_classes=80,
                 warmup_epoch=4,
                 img_size=640,
                 iou_type='giou',
                 loss_weight=None,
                 feats_shape=None):
        super(ComputeLoss, self).__init__()
        if fpn_strides is None:
            fpn_strides = [8, 16, 32]
        if loss_weight is None:
            loss_weight = {
                "class": 1.0,
                "box": 2.5
            }
        if feats_shape is None:
            feats_shape = [(80, 80), (40, 40), (20, 20)]
        self.img_size = img_size
        self.num_classes = num_classes
        self.warmup_epoch = warmup_epoch
        self.iou_type = iou_type
        self.loss_weight = loss_weight
        self.fpn_strides = fpn_strides
        if warmup_epoch > 0:
            self.atss_assigner = ATSSAssigner(topk=9, num_classes=num_classes)
        self.task_align_assigner = TaskAlignedAssigner(topk=13, num_classes=num_classes)

        self.varifocal_loss = VarifocalLoss()

        self.box_loss = BoxLoss(iou_type=iou_type)

    def forward(self, outputs, targets, mask_gt, epoch_num):
        # cls_pred (bs, num_total_anchors, num_classes)
        # box_pred (bs, num_total_anchors, 4)
        cls_pred, box_pred = outputs

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchor(self.img_size, self.fpn_strides, device="cuda")
        batch_size = cls_pred.size(0)
        # targets = self.preprocess(targets, batch_size)
        gt_labels = targets[:, :, 0]
        gt_boxes = targets[:, :, 1:]

        # pred_boxes
        anchor_point_s = anchor_points / stride_tensor
        pred_boxes = self.box_decode(anchor_point_s, box_pred, stride_tensor)

        # assigner
        if epoch_num < self.warmup_epoch:
            target_labels, target_boxes, target_scores, candidate_id = \
                self.atss_assigner(anchors,
                                   num_anchors_list,
                                   gt_labels,
                                   gt_boxes,
                                   mask_gt,
                                   pred_boxes)
        else:
            target_labels, target_boxes, target_scores, candidate_id = \
                self.task_align_assigner(cls_pred,
                                         pred_boxes * stride_tensor,
                                         gt_labels,
                                         gt_boxes,
                                         mask_gt)
        targets_labels = target_labels.cuda()
        targets_boxes = target_boxes.cuda()
        targets_scores = target_scores.cuda()
        candidate_id = candidate_id.cuda()

        targets_boxes /= stride_tensor

        label = F.one_hot(targets_labels.long(), self.num_classes + 1)[..., :-1]

        scores_sum = targets_scores.sum()
        if scores_sum > 0:
            # 分类损失
            loss_cls = self.varifocal_loss(cls_pred, targets_scores, label)
        else:
            loss_cls = 0
        # 除以应该正确预测的所有值的综合(被iou加权了的)
        if scores_sum > 1:
            loss_cls /= scores_sum
        box_loss = self.box_loss(pred_boxes, targets_boxes, targets_scores, candidate_id)
        loss = self.loss_weight["class"] * loss_cls + self.loss_weight["box"] * box_loss
        return loss, self.loss_weight["class"] * loss_cls, self.loss_weight["box"] * box_loss

    def box_decode(self, anchor_points, box_pred, stride_tensor):
        """
        解码预测框
        """

        box_pred = torch.cat([anchor_points.unsqueeze(0).repeat(box_pred.size(0), 1, 1)
                              - box_pred[..., :2],
                              box_pred[..., 2:] +
                              anchor_points.unsqueeze(0).repeat(box_pred.size(0), 1, 1)], dim=-1)
        return box_pred


class VarifocalLoss(nn.Module):
    """
    分类损失
    """

    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """
        expression: VFL(p,q)=\left\{\begin{matrix}-q(qlog(p)+(1-q)log(1-p)),&q>0
        \\-\alpha p^\gamma log(1-p),&q=0\end{matrix}\right
        """
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        return (F.binary_cross_entropy(pred_score, gt_score.to(pred_score), reduction="none") * weight).sum()


class BoxLoss(nn.Module):

    def __init__(self, iou_type='giou'):
        super(BoxLoss, self).__init__()
        self.iou_loss = IOULoss(box_format="xyxy", iou_type=iou_type, eps=1e-9)

    def forward(self, pred_boxes, target_boxes, target_scores, mask):
        num_pos = mask.sum()

        if num_pos > 0:
            # 展平
            mask = mask.flatten()
            pred_boxes = pred_boxes.reshape(-1, 4)
            target_boxes = target_boxes.reshape(-1, 4)
            target_scores = target_scores.sum(-1).flatten()
            # TODO 预测的box和目标的box对应不起来
            pred_boxes_pos = pred_boxes[mask, :]
            target_boxes_pos = target_boxes[mask, :]
            box_weight = target_scores[mask]
            loss_box = self.iou_loss(pred_boxes_pos, target_boxes_pos).squeeze(-1) * box_weight
            target_scores_sum = target_scores.sum()
            if target_scores_sum > 1:
                loss_box = loss_box.sum() / target_scores_sum
            else:
                loss_box = loss_box.sum()
        else:
            loss_box = torch.tensor(0.0).to(pred_boxes.device)
        return loss_box
