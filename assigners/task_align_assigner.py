"""
@author supermantx
@date 2024/7/9 14:10
TaskAlignedAssigner
"""
import torch
from torch import nn
from torch.nn import functional as F

from yolo.util.box_util import iou_calculator, select_candidates_in_gts


class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_boxes, gt_labels, gt_boxes, mask_gt):
        """
        pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
        pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
        anc_points (Tensor): shape(num_total_anchors, 2)
        gt_labels (Tensor): shape(bs, n_max_boxes, 1)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_boxes.size(1)
        self.n_anchors = pd_scores.size(1)
        overlaps = []
        for i in range(self.bs):
            overlap = iou_calculator(gt_boxes[i], pd_boxes[i])
            overlaps.append(overlap)
        overlaps = torch.stack(overlaps, dim=0)
        pd_scores = pd_scores.argmax(-1).squeeze(-1)
        pd_scores = pd_scores.unsqueeze(1).repeat(1, self.n_max_boxes, 1)
        alignment_metric = pd_scores.pow(self.alpha) * overlaps.pow(self.beta)
        in_gts = []
        for i in range(self.bs):
            in_gt = select_candidates_in_gts(pd_boxes[i], gt_boxes[i])
            in_gts.append(in_gt)
        in_gts = torch.stack(in_gts, dim=0)
        # select topk candidates
        alignment_metric = in_gts * alignment_metric * mask_gt
        _, indices = alignment_metric.topk(self.topk, dim=-1)
        candidate_id = F.one_hot(indices, num_classes=self.n_anchors).sum(-2)
        # solve the problem of multiple gt_boxes matching the same anchor_box
        multi_gt_match = candidate_id.sum(-2) > 1
        multi_gt_match = multi_gt_match.unsqueeze(dim=1)
        multi_gt_match_iou = overlaps * multi_gt_match.repeat([1, self.n_max_boxes, 1])
        candidate_id = candidate_id * ~multi_gt_match.repeat((1, self.n_max_boxes, 1))
        values, indices = multi_gt_match_iou.max(1)
        candidate_id = F.one_hot(indices, self.n_max_boxes).permute(0, 2, 1).bool() * multi_gt_match.repeat(
            (1, self.n_max_boxes,
             1)) + candidate_id
        target_labels, target_boxes, target_scores, mask = self.get_targets(gt_labels, gt_boxes, candidate_id)

        # TODO normalize alignment_metric
        return target_labels, target_boxes, target_scores, mask

    def get_targets(self, gt_labels, gt_boxes, candidate_id):
        """
        gt_labels (Tensor): shape(bs, n_max_boxes, 1)
        gt_boxes (Tensor): shape(bs, n_max_boxes, 4)
        candidate_id (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        """
        idx = candidate_id.permute(0, 2, 1)
        mask = idx.sum(-1) > 0
        target_labels = torch.where(mask, gt_labels[
            torch.arange(self.bs).unsqueeze(1).repeat(1, self.n_anchors).flatten(0), idx.argmax(-1).flatten(0)].reshape(
            self.bs, self.n_anchors), torch.full(mask.shape, self.bg_idx))

        # assigned target boxes
        target_idx = idx.argmax(-1).squeeze() + torch.arange(self.bs).unsqueeze(1).repeat(1, self.n_anchors) * self.n_max_boxes
        target_boxes = gt_boxes.reshape([-1, 4])[target_idx]
        # assigned target scores
        target_scores = F.one_hot(target_labels, self.num_classes + 1).float()
        target_scores = target_scores[:, :, :self.num_classes]
        return target_labels, target_boxes, target_scores, mask


if __name__ == '__main__':
    assigner = TaskAlignedAssigner()
    pt_labels = torch.randint(0, 80, [2, 8400, 80])
    pt_bboxes = torch.rand([2, 8400, 4]) * 640
    gt_bboxes = torch.rand([2, 3, 4]) * 640
    gt_labels = torch.randint(0, 80, [2, 3])
    mask_gt = torch.ones([2, 3, 8400])
    pd_bboxes = torch.rand([2, 3, 4])
    target_labels, target_boxes, target_scores, candidate_id = assigner(pt_labels, pt_bboxes, gt_labels, gt_bboxes, mask_gt)
