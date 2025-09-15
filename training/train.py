import os
import random
from datetime import datetime


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import segmentation_models_pytorch as smp
import torchmetrics
import torchmetrics.segmentation
import hydra
import wandb


from custom_dataset import SegmentationDataset
from data.preprocessed.train_test_split import train_imgs, val_imgs, train_masks, val_masks


"""
Training script for image segmentation using PyTorch and segmentation_models_pytorch.

This script sets up a segmentation dataset, defines a model (UNet or DeepLabV3+),
trains the model, evaluates it on a validation set, logs metrics to Weights & Biases (wandb),
and saves the trained model.

Workflow:
1. Set random seeds for reproducibility.
2. Define image and mask transformations.
3. Prepare training and validation datasets using SegmentationDataset.
4. Create DataLoaders for batch processing.
5. Initialize the model, optimizer, loss function, and evaluation metrics.
6. Train the model for a specified number of epochs:
   - Perform training for one epoch.
   - Evaluate on the validation set.
   - Log losses and metrics to console and wandb.
   - Reset metrics for the next epoch.
7. Save the trained model to the specified directory and log with wandb.

Key Features:
- Supports UNet and DeepLabV3+ architectures with configurable encoder.
- Uses DiceLoss for multi-class segmentation.
- Tracks Accuracy, IoU, and Dice score using torchmetrics.
- Full reproducibility with seed setting for CPU and CUDA.
- Integration with Hydra for configuration management and wandb for experiment tracking.

Dependencies:
    - torch, torchvision, numpy, random
    - segmentation_models_pytorch
    - torchmetrics
    - hydra-core
    - wandb
    - PIL (Pillow)
"""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

    

def get_model(cfg):
    if cfg.model.architecture == "unet":
        return smp.Unet(
            encoder_name=cfg.model.encoder,
            in_channels=3,
            classes=cfg.num_classes
        )
    elif cfg.model.architecture == "deeplabv3+":
        return smp.DeepLabV3Plus(
            encoder_name=cfg.model.encoder,
            in_channels=3,
            classes=cfg.num_classes
        )
    else:
        raise ValueError(f"Unknown model {cfg.model.architecture}")

   
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, loss_fn, metrics, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            for metric in metrics.values():
                metric.update(preds, masks)
    return total_loss / len(loader), {k: m.compute().item() for k, m in metrics.items()}    


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    wandb.init(
        project="segmentation_project", 
        config={
            "architecture": cfg.model.architecture,
            "epochs": cfg.train.epochs,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr,
            "img_size": cfg.dataset.img_size,
            "num_classes": cfg.num_classes
        },                   
        name=f"{cfg.model.architecture}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    )

    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    
    img_transform = transforms.Compose([
        transforms.Resize(tuple(cfg.dataset.img_size)),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(tuple(cfg.dataset.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: (x > 0).long().squeeze())
    ])


    train_dataset = SegmentationDataset(
        train_imgs, train_masks, 
        transform=img_transform, 
        mask_transform=mask_transform
    )
    val_dataset = SegmentationDataset(
        val_imgs, val_masks, 
        transform=img_transform, 
        mask_transform=mask_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = get_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, classes=cfg.num_classes)


    metrics = {
        "accuracy": torchmetrics.Accuracy(task="binary").to(device),
        "iou": torchmetrics.JaccardIndex(task="binary").to(device),
        "dice": torchmetrics.segmentation.DiceScore(num_classes=cfg.num_classes, average="macro").to(device)
    }

    os.makedirs(cfg.train.save_dir_model, exist_ok=True)

    for epoch in range(cfg.train.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, metrics, device)

        print(
            f"Epoch {epoch+1}/{cfg.train.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | Dice: {val_metrics['dice']:.4f}"
        )

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics['accuracy'],
            "val_iou": val_metrics['iou'],
            "val_dice": val_metrics['dice'],
            "epoch": epoch + 1
        })

        for m in metrics.values():
            m.reset()

    save_path = f"{cfg.train.save_dir_model}/{cfg.model.architecture}_custom.pth"
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path) 


if __name__ == "__main__":
    main()


