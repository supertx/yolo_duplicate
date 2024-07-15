"""
@author supermantx
@date 2024/7/3 16:51
读取coco数据集
yolo做的是目标检测，所以只要框的坐标和类别即可，
yolov6源码对coco的数据集进行了重新整理
文件结构
coco
    annotations
        instances_train2017.json
        instances_val2017.json
    images
        train2017
        val2017
    labels
        train2017
        val2017
"""
import yaml
import os.path as osp
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F

from yolo_duplicate.util.box_util import cxcywh2xyxy


class CocoDataset(Dataset):

    def __init__(self, root_dir, yml_file, preload=False, train=True, transform=None):
        super(CocoDataset, self).__init__()
        self.preload = preload
        self.root_dir = root_dir
        self.train = train
        if train:
            self.img_dir = osp.join(root_dir, "images/train2017")
            self.ann_file = osp.join(root_dir, "labels/train2017")

        else:
            self.img_dir = osp.join(root_dir, "images/val2017")
            self.ann_file = osp.join(root_dir, "labels/val2017")
        if not transform:
            self.transform = get_transform()
        else:
            self.transform = transform
        self.img_name_list = []
        self.ann_name_list = []
        self.img_list = []
        # 检查图片和标签文件
        self.check_files()
        # 预加载图片和标签
        if preload:
            self.load_data()
        load = yaml.full_load(open(yml_file))
        self.class_names = load["names"]

    def __getitem__(self, index):
        if self.preload:
            img = self.img_list[index]
        else:
            img = cv.imread(osp.join(self.img_dir, self.img_name_list[index]))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB, img)
        annos = self.read_ann(self.ann_name_list[index], img.shape)
        if self.train:
            # TODO 图像增强
            # 图像增强 Mosaic Augmentation 是将四张图片拼到一块
            # 随机仿射变换
            pass
        boxes = self.boxes_transform(annos[1][:, 1:], img.shape, (640, 640))
        annos[1][:, 1:] = boxes
        img = self.transform(img)
        return img, torch.tensor(annos[1])

    def draw_box(self, img, annos, mask):
        if isinstance(img, torch.Tensor):
            img = img.detach().numpy().transpose(1, 2, 0)
            img = img * 255
            img = img.astype(np.uint8)
            # 图片必须连续存储
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)
        for anno, m in zip(annos, mask):
            if not m:
                break
            if isinstance(anno, torch.Tensor):
                anno = anno.numpy()
            box = anno[1:5]
            cls = anno[0].astype(np.int32)
            box = cxcywh2xyxy(box)
            box = box.astype(np.int32)
            cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv.putText(img, self.class_names[cls], (box[0], box[1] - 10 if box[1] > 10 else 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        plt.imshow(img)
        plt.show()

    def __len__(self):
        # return len(self.img_name_list)
        return 1000

    def boxes_transform(self, boxes, ori_shape, img_shape):
        ret_box = []
        for box in boxes:
            ret_box.append([box[0] / ori_shape[1] * img_shape[1],
                            box[1] / ori_shape[0] * img_shape[0],
                            box[2] / ori_shape[1] * img_shape[1],
                            box[3] / ori_shape[0] * img_shape[0]])
        return np.array(ret_box)

    def read_ann(self, ann_file, img_shape):
        with open(osp.join(self.ann_file, ann_file), "r") as f:
            annos = f.readlines()
        annos = np.array([anno.strip().split(" ") for anno in annos], dtype=float)
        annos_box = annos[:, 1:5]
        annos_cls = annos[:, 0]
        annos_box[:, 0] = annos_box[:, 0] * img_shape[1]
        annos_box[:, 1] = annos_box[:, 1] * img_shape[0]
        annos_box[:, 2] = annos_box[:, 2] * img_shape[1]
        annos_box[:, 3] = annos_box[:, 3] * img_shape[0]
        annos_processed = np.full_like(annos, 0)
        annos_processed[:, 0] = annos_cls
        annos_processed[:, 1:5] = annos_box
        return annos, annos_processed

    def check_files(self):
        assert osp.exists(self.img_dir) and osp.exists(
            self.ann_file), f"img dir {self.img_dir} or annotation file {self.ann_file} doesn't exists"
        for _, _, filenames in os.walk(self.img_dir):
            if len(filenames):
                t = tqdm(filenames, desc="Checking files... valid img count: (0/0)", ncols=150)
                for f in t:
                    if osp.isfile(osp.join(self.ann_file, f.replace("jpg", "txt"))):
                        self.img_name_list.append(f)
                        self.ann_name_list.append(f.replace("jpg", "txt"))
                        t.desc = f"Checking files... valid img count:  ({len(self.img_name_list)}/{t.n if t.n > 0 else len(t)})"

    def load_data(self):
        """
        将图片和标签加载到内存中，训练集有19g，测试集788MB
        :return:
        """
        pass


def collate_fn(batch):
    imgs, annos = zip(*batch)
    max_batch_box = max([anno.shape[0] for anno in annos])
    annos_list = []
    mask = []
    for anno in annos:
        annos_list.append(F.pad(anno, (0, 0, 0, max_batch_box - anno.shape[0]), value=0))
        mask.append(torch.tensor([1] * anno.shape[0] + [0] * (max_batch_box - anno.shape[0])))
    return (torch.stack(imgs, dim=0),
            torch.stack(annos_list, dim=0),
            torch.stack(mask, dim=0))


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def dataloader(batch_size, root_dir, yml_file, train=True):
    coco = CocoDataset(root_dir, yml_file, train=train)
    return DataLoader(coco, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
