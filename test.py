"""
@author supermantx
@date 2024/7/23 13:47
测试训练出来的模型
"""
import numpy as np
import torch
import cv2 as cv
from matplotlib import pyplot as plt

from dataset import get_transform
from model import YOLOV6
from assigners import generate_anchor

from yolo_duplicate.models.yolo import build_model
from yolo_duplicate.util.config import Config

if __name__ == '__main__':
    c = Config.fromfile("config/yolov6n.py")
    if not hasattr(c, 'training_mode'):
        setattr(c, 'training_mode', 'repvgg')
    model = build_model(c, 80, torch.device('cuda'), False, False)
    model.load_state_dict(torch.load("logs/model_12.pt"))
    # model.eval()
    model.cuda()
    output = None
    img = cv.imread("/data/tx/coco/images/train2017/000000504378.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # 处理图片
    input = get_transform()(img)
    with torch.no_grad():
        _, cls_pred, box_pred = model(input.unsqueeze(0).cuda())[0]
        max_pred = cls_pred.max(-1)[0]
        values, indices = max_pred.topk(5, dim=-1)
        max_id = torch.nn.functional.one_hot(indices, num_classes=8400).sum(-2)
        _, anchor_points, _, stride_tensor = generate_anchor(640, [8, 16, 32], device="cuda")
        anchor_points = anchor_points / stride_tensor
        anchor_points = anchor_points.unsqueeze(0)
        pred_points = torch.stack([anchor_points[..., 0] - box_pred[..., 0], anchor_points[..., 1] - box_pred[..., 1],
                                   anchor_points[..., 0] + box_pred[..., 2], anchor_points[..., 1] + box_pred[..., 3]], dim=-1)
        pred_points *= stride_tensor
        pred_points = torch.masked_select(pred_points.squeeze(0), max_id.bool().squeeze(0).unsqueeze(-1).repeat([1, 4])).reshape([-1, 4])
    for i in range(pred_points.shape[0]):
        img = cv.rectangle(img, pred_points[i, :2].cpu().numpy().astype(np.int32), pred_points[i, -2:].cpu().numpy().astype(np.int32), (0, 255, 0), 2)
    plt.imshow(img)
    plt.show()