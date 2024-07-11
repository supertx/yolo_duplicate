"""
@author supermantx
@date 2024/7/4 14:39
"""
import numpy as np
import torchvision.ops.boxes as box_ops
import torch


def cxcywh2xyxy(boxes):
    if len(boxes.shape) == 1:
        return np.array(
            [boxes[0] - boxes[2] / 2, boxes[1] - boxes[3] / 2, boxes[0] + boxes[2] / 2, boxes[1] + boxes[3] / 2])
    ret = []
    for i, box in enumerate(boxes):
        ret.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
    return np.array(ret)


def xyxy2cxcywh(boxes):
    if len(boxes.shape) == 1:
        return np.array(
            [(boxes[0] + boxes[2]) / 2, (boxes[1] + boxes[3]) / 2, boxes[2] - boxes[0], boxes[3] - boxes[1]])
    ret = []
    for i, box in enumerate(boxes):
        ret.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])
    return np.array(ret)


def xywh2xyxy(boxes):
    if len(boxes.shape) == 1:
        return np.array([boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]])
    ret = []
    for i, box in enumerate(boxes):
        ret.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
    return np.array(ret)


def xyxy2xywh(boxes):
    if len(boxes.shape) == 1:
        return np.array([boxes[0], boxes[1], boxes[2] - boxes[0], boxes[3] - boxes[1]])
    ret = []
    for i, box in enumerate(boxes):
        ret.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
    return np.array(ret)


def iou_calculator(boxes1, boxes2, mode='iou'):
    """
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    if mode == 'iou':
        return box_ops.box_iou(boxes1, boxes2)
    elif mode == 'giou':
        return box_ops.generalized_box_iou(boxes1, boxes2)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def dist_calculator(boxes1, boxes2):
    """
    calculate the distance between boxes1 and boxes2
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise distance values for every element in boxes1 and boxes2
    """
    boxes1_c = torch.stack([boxes1[:, 0] + boxes1[:, 2] / 2, boxes1[:, 1] + boxes1[:, 3] / 2], dim=1)
    boxes2_c = torch.stack([boxes2[:, 0] + boxes2[:, 2] / 2, boxes2[:, 1] + boxes2[:, 3] / 2], dim=1)

    dist = torch.sqrt((boxes1_c[:, None, 0] - boxes2_c[None, :, 0]) ** 2
                      + (boxes1_c[:, None, 1] - boxes2_c[None, :, 1]) ** 2)
    return dist


def select_candidates_in_gts(anchor_boxes, gt_boxes, eps=1e-9):
    """
    select the positive anchors center in gt
    """
    ac_boxes_center = torch.stack([(anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2,
                                   (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2], dim=1)
    bs = 0
    if len(gt_boxes.shape) == 2:
        n_max_boxes = gt_boxes.size(0)
    else:
        bs, n_max_boxes, _ = gt_boxes.shape
    ac_boxes_center.repeat(1, n_max_boxes, 1)
    ac_boxes_center = ac_boxes_center.reshape(-1, 2)
    gt_boxes = gt_boxes.reshape(-1, 4)
    gt_boxes_lt = gt_boxes[:, :2].unsqueeze(1).repeat(1, anchor_boxes.size(0), 1)
    gt_boxes_rb = gt_boxes[:, 2:].unsqueeze(1).repeat(1, anchor_boxes.size(0), 1)
    b_lt = ac_boxes_center - gt_boxes_lt
    b_rb = gt_boxes_rb - ac_boxes_center
    bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
    if bs == 0:
        return bbox_deltas.min(dim=-1)[0] > eps
    else:
        bbox_deltas = bbox_deltas.reshape(bs, n_max_boxes, anchor_boxes.size(0), -1)
    return bbox_deltas.min(dim=-1)[0] > eps
