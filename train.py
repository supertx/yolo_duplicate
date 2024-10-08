"""
@author supermantx
@date 2024/7/12 17:10
训练yolov6
"""
import sys
import os
sys.path.append(os.getcwd())
print(sys.path)
from datetime import datetime


from tqdm import tqdm
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt

from dataset import get_dataloader
from util.config_util import load_yaml
from model import YOLOV6
from yolo_duplicate.criterion.loss import ComputeLoss
from yolo_duplicate.models.yolo import build_model
from yolo_duplicate.util.config import Config

def train_one_epoch(cfg, epoch, model, loss, dataloader, optimizer, device):
    model.train()
    t = tqdm(dataloader, desc=f"Epoch ({epoch}/{cfg.EPOCHS}) loss : 0.0", ncols=150)
    for i, (images, annos, mask) in enumerate(t):
        images = images.to(device)
        annos = annos.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        (l, cls_l, iou_l) = loss(outputs, annos, mask, epoch)
        l.backward()
        t.desc = f"Epoch ({epoch}/{cfg.EPOCHS}) loss : {l.item():.2f} cls_loss : {cls_l.item():.2f} iou_loss : {iou_l.item():.2f}"
        optimizer.step()


def train(args):
    cfg = load_yaml(args.config)
    dataloader = get_dataloader(cfg, train=True)
    test_dataloader = get_dataloader(cfg, train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compute_loss = ComputeLoss(warmup_epoch=cfg.WARMUP_EPOCH)
    # model = YOLOV6()
    c = Config.fromfile("config/yolov6n.py")
    if not hasattr(c, 'training_mode'):
        setattr(c, 'training_mode', 'repvgg')
    model = build_model(c, 80, torch.device('cuda'), False, False)
    model.to(device)
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)

    optimizer = torch.optim.Adam(g_bnw, lr=0.01, betas=(0.8, 0.999))
    optimizer.add_param_group({'params': g_w, 'weight_decay': 0.001})
    optimizer.add_param_group({'params': g_b})
    # optimizer = torch.optim.Adam(model.get_learnable_parameter(), lr=cfg.LR)
    schedule_lr = StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    epoch = 0

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    logfile = os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.EXPERIMENT_NAME}_{datetime.now().strftime('%Y%m%d%H%M')}.log")
    if cfg.RESUME:
        checkpoint = torch.load(cfg.CHECKPOINT)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        schedule_lr.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
        del checkpoint

    best_accuracy = 0
    for epoch in range(epoch, cfg.EPOCHS):
        train_one_epoch(cfg, epoch, model, compute_loss, dataloader, optimizer, device)
        schedule_lr.step()

        if epoch % cfg.SAVE_INTERVAL == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": schedule_lr.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, os.path.join(cfg.CHECKPOINT_DIR, f"checkpoint_{epoch}.pt"))
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"model_{epoch}.pt"))
            del checkpoint

            accuracy = compute_accuracy(model, test_dataloader, device)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            print(f"Epoch {epoch}, accuracy: {accuracy}, best accuracy: {best_accuracy}")


@torch.no_grad()
def compute_accuracy(model, test_dataloader, device):
    return 0.0
    model.eval()
    t = tqdm(test_dataloader, desc="acc : 0.0")
    for i, (images, annos, mask) in enumerate(t):
        images = images.to(device)
        annos = annos.to(device)
        mask = mask.to(device)

        outputs = model(images)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOV6")
    parser.add_argument("--config", type=str, default="config/yolo6n.yaml", help="path to config file")
    args = parser.parse_args()
    train(args)
    print("training process finished")


if __name__ == '__main__':
    main()
