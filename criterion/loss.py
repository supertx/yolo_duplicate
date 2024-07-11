"""
@author supermantx
@date 2024/7/4 15:49
计算loss
"""

import torch
from torch import nn
from torch.nn import functional as F

from yolo.assigners import ATSSAssigner, TaskAlignedAssigner, generate_anchor


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
                "box": 1.0
            }
        if feats_shape is None:
            feats_shape = [(80, 80), (40, 40), (20, 20)]
        self.img_size = img_size
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.warmup_epoch = warmup_epoch
        self.iou_type = iou_type
        self.loss_weight = loss_weight
        self.fpn_strides = fpn_strides
        if warmup_epoch > 0:
            self.atss_assigner = ATSSAssigner(topk=9, num_classes=num_classes)
        self.task_align_assigner = TaskAlignedAssigner(topk=13, num_classes=num_classes)

    def forward(self, outputs, targets, mask_gt, epoch_num):
        # cls_pred (bs, num_total_anchors, num_classes)
        # box_pred (bs, num_total_anchors, 4)
        cls_pred, box_pred = outputs

        anchors, anchor_points, num_anchors_list, stride_tensor = generate_anchor(self.img_size, self.fpn_strides)
        batch_size = cls_pred.size(0)
        targets = self.preprocess(targets, batch_size)
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
                                         pred_boxes,
                                         gt_labels,
                                         gt_boxes,
                                         mask_gt)
        targets_labels = target_labels.cuda()
        targets_boxes = target_boxes.cuda()
        targets_scores = target_scores.cuda()
        candidate_id = candidate_id.cuda()

        targets_boxes /= stride_tensor

    def box_decode(self, anchor_points, box_pred, stride_tensor):
        """
        解码预测框
        """
        box_pred = torch.cat([box_pred[..., :2] +
                              anchor_points.unsqueeze(0).repeat(box_pred.size(0), 1, 1),
                              box_pred[..., 2:] +
                              anchor_points.unsqueeze(0).repeat(box_pred.size(0), 1, 1)], dim=-1)
        box_pred = box_pred * stride_tensor
        return box_pred


class VarifocalLoss(nn.Module):
    """
    分类损失
    """
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):