"""
@author supermantx
@date 2024/7/12 17:16
"""

import os

from torch.utils.data import DataLoader

from .coco import dataloader as coco_dataloader
from yolo_duplicate.util.config_util import load_yaml

__all__ = ["get_dataloader"]

datasets = ["coco"]


def get_dataloader(cfg, yml_file, train=True, dataset="coco"):
    if dataset not in datasets:
        raise ValueError(f"dataset {dataset} not supported")
    if dataset == "coco":
        return coco_dataloader(cfg.BATCH_SIZE, cfg.ROOT_DIR, yml_file, train)
