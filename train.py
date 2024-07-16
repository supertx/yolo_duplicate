"""
@author supermantx
@date 2024/7/12 17:10
训练yolov6
"""

import os
from datetime import datetime

from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR

from dataset import get_dataloader
from util.config_util import load_yaml
from model import YOLOV6
from yolo_duplicate.criterion.loss import ComputeLoss


def train_one_epoch(cfg, epoch, model, loss, dataloader, optimizer, device):
    model.train()
    t = tqdm(dataloader, desc=f"Epoch ({epoch}/{cfg.EPOCHS}) loss : 0.0")
    for i, (images, annos, mask) in enumerate(t):
        images = images.to(device)
        annos = annos.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        (loss, _, _) = loss(outputs, annos, mask, epoch)
        loss.backward()
        t.desc = f"Epoch ({epoch}/{cfg.EPOCHS}) loss : {loss.item():.2f}"
        optimizer.step()


def train(args):
    cfg = load_yaml(args.config)
    dataloader = get_dataloader(cfg, train=True)
    test_dataloader = get_dataloader(cfg, train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compute_loss = ComputeLoss()
    model = YOLOV6()
    model.to(device)
    optimizer = torch.optim.Adam(model.get_learnable_parameter(), lr=cfg.LR)
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
