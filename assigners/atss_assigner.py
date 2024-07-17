"""
@author supermantx
@date 2024/7/4 15:46
ATSS_assigner
"""
import torch
from torch import nn
from torch.nn import functional as F
from yolo_duplicate.util.box_util import iou_calculator, dist_calculator, select_candidates_in_gts


class ATSSAssigner(nn.Module):
    """
    Adaptive Training Sample Selection Assigner
    """

    def __init__(self, topk=9, num_classes=80):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes

    @torch.no_grad()
    def forward(self, anc_bboxes, n_level_bboxes, gt_labels, gt_bboxes, mask_gt, pd_boxes):
        """
        anc_bboxes (Tensor): shape(num_total_anchors, 4)
        gt_bbox (Tensor): shape(bs, n_max_boxes, 4)
        """
        self.n_anchors = anc_bboxes.size(0)
        self.bs = gt_bboxes.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        mask_gt = mask_gt.unsqueeze(-1).repeat(1, 1, self.n_anchors)
        overlaps = iou_calculator(anc_bboxes, gt_bboxes.reshape([-1, 4]))
        distance = dist_calculator(anc_bboxes, gt_bboxes.reshape([-1, 4]))
        overlaps = overlaps.reshape(self.bs, self.n_max_boxes, self.n_anchors)
        distance = distance.reshape(self.bs, self.n_max_boxes, self.n_anchors)
        candidate_id, candidate_counts = self.select_topk_candidates(distance, overlaps, n_level_bboxes, mask_gt)
        # candidate anchor box center should be in gt box
        is_in_gts = select_candidates_in_gts(anc_bboxes, gt_bboxes)
        candidate_id =  candidate_id * mask_gt
        # print(candidate_id.sum())

        target_labels, target_boxes, target_scores, mask_gt = self.get_targets(gt_labels, gt_bboxes, candidate_id)
        # soft label with iou
        if pd_boxes is not None:
            iou = []
            for i in range(self.bs):
                iou.append(iou_calculator(pd_boxes[i], gt_bboxes[i]))
            iou = torch.stack(iou, dim=0)
            iou = iou.max(-1)[0].squeeze(-1) * mask_gt
            target_scores = iou.unsqueeze(-1).repeat(1, 1, self.num_classes) * target_scores
        # target_scores = torch.abs(target_scores)
        return target_labels, target_boxes, target_scores, mask_gt

    def select_topk_candidates(self, distances, overlaps, n_level_bboxes, mask_gt):
        distances *= mask_gt
        start_idx = 0
        indices_lst = []
        # calculate the tok distance for each level
        for level_counts in n_level_bboxes:
            values, indices = distances[:, :, start_idx:start_idx + level_counts].topk(self.topk, dim=-1, largest=False)
            indices_lst.append(indices + start_idx)
            start_idx += level_counts
        indices = torch.cat(indices_lst, dim=-1)
        candidate_id = F.one_hot(indices, num_classes=n_level_bboxes.sum()).sum(-2)
        # print(candidate_id.sum())
        """
        official implement have make a emit for a anchor_box matching multiple gt_boxes first, and select these omitted
         anchor_box after calculate iou condition(select the highest iou with gt).
        """
        multi_gt_match = candidate_id.sum(-2) > 1
        multi_gt_match = multi_gt_match.unsqueeze(dim=1)
        multi_gt_match_iou = overlaps * multi_gt_match.repeat([1, self.n_max_boxes, 1])
        candidate_id = candidate_id * ~multi_gt_match.repeat((1, self.n_max_boxes, 1))
        values, indices = multi_gt_match_iou.max(1)
        candidate_id = F.one_hot(indices, self.n_max_boxes).permute(0, 2, 1).bool() * multi_gt_match.repeat(
            (1, self.n_max_boxes,
             1)) + candidate_id
        # calculate the average iou and std iou
        avg_iou = (overlaps * candidate_id).sum(2) / candidate_id.sum(2)
        std_iou = (overlaps * candidate_id).std(2)
        # select anchor box where iou > avg_iou + std_iou
        candidate_id = torch.where(overlaps > (avg_iou + std_iou).unsqueeze(2).repeat(1, 1, self.n_anchors),
                                   candidate_id,
                                   torch.zeros_like(candidate_id))
        # print(candidate_id.sum())
        return candidate_id, candidate_id.sum(1)

    def get_targets(self, gt_labels, gt_boxes, target_gt_idx):
        # assigned target labels
        idx = target_gt_idx.permute(0, 2, 1)
        mask = idx.sum(-1) > 0
        target_labels = torch.where(mask, gt_labels[
            torch.arange(self.bs).unsqueeze(1).repeat(1, self.n_anchors).flatten(0).to(mask.device), idx.argmax(-1).flatten(0)].reshape(
            self.bs, self.n_anchors), torch.full(mask.shape, self.bg_idx).to(mask.device))

        # assigned target boxes
        target_idx = idx.argmax(-1).squeeze() + torch.arange(self.bs).unsqueeze(1).repeat(1, self.n_anchors).to(idx.device) * self.n_max_boxes
        target_boxes = gt_boxes.reshape([-1, 4])[target_idx]
        # assigned target scores
        target_scores = F.one_hot(target_labels.long(), self.num_classes + 1).float()
        target_scores = target_scores[:, :, :self.num_classes]
        return target_labels, target_boxes, target_scores, mask


if __name__ == '__main__':
    from anchor_generator import generate_anchor

    atss_assigner = ATSSAssigner()
    anchor = generate_anchor(640, [8, 16, 32])[0]

    gt_bboxes = torch.rand([2, 3, 4]) * 640
    gt_labels = torch.randint(0, 80, [2, 3])
    mask_gt = torch.ones([2, 3, 8400])
    pd_bboxes = torch.rand([2, 3, 4])
    target_labels, target_boxes, target_scores, candidate_id = atss_assigner(anchor, torch.tensor([6400, 1600, 400]), gt_labels, gt_bboxes, mask_gt, pd_bboxes)
