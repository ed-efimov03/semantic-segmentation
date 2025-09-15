"""
Segmentation Model Evaluation Script.

This script loads a pretrained segmentation model (UNet or DeepLabV3+) and evaluates it on a 
test dataset. It supports custom datasets via the `SegmentationDataset` class and uses 
`segmentation_models_pytorch` for model architectures and loss functions. Metrics such as 
accuracy, IoU, and Dice score are computed using `torchmetrics`.

The script also logs evaluation results to Weights & Biases (wandb).

Modules:
    os: For file path operations.
    datetime: To timestamp model runs.
    hydra: For configuration management.
    torch: Core PyTorch functionality.
    torchvision.transforms: Image preprocessing.
    numpy: For reproducibility and array operations.
    segmentation_models_pytorch (smp): For model architectures and loss functions.
    torchmetrics: For evaluation metrics.
    wandb: For logging results.
    custom_dataset: User-defined dataset class for segmentation.

Functions:
    set_seed(seed):
        Sets the random seed for reproducibility across numpy, torch, and CUDA.

    get_model(cfg):
        Returns a segmentation model (UNet or DeepLabV3+) based on configuration.

    evaluate(model, loader, loss_fn, metrics, device):
        Evaluates the model on a given DataLoader and returns loss and computed metrics.

Main Execution:
    Uses Hydra to load configuration and initialize model, dataset, transformations, 
    metrics, and logging. Evaluates the model on the test dataset and logs results.
"""
    
import os
from datetime import datetime
import hydra

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import segmentation_models_pytorch as smp
import torchmetrics
import wandb
import torchmetrics.segmentation

from custom_dataset import SegmentationDataset


def set_seed(seed):
    import random
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



image_dir = "data/test/Histo-Seg-H&E-Whole-Slide-Image-Segmentation-Dataset/Histo-Seg/Images-prep"
mask_dir  = "data/test/Histo-Seg-H&E-Whole-Slide-Image-Segmentation-Dataset/Histo-Seg/Masks-prep"

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")
if not os.path.exists(mask_dir):
    raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

images = sorted(os.listdir(image_dir))
masks  = sorted(os.listdir(mask_dir))

image_paths = [os.path.join(image_dir, img) for img in images]
mask_paths  = [os.path.join(mask_dir, msk) for msk in masks]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    if cfg.model.architecture == "unet":
        model_path = "models/unet_custom.pth"
    elif cfg.model.architecture == "deeplabv3+":
        model_path = "models/deeplabv3+_custom.pth"

    wandb.init(
        project="segmentation_project_test",
        config={
            "architecture": cfg.model.architecture,
            "batch_size": cfg.train.batch_size,
            "img_size": cfg.dataset.img_size,
            "num_classes": cfg.num_classes
        },
        name=f"{cfg.model.architecture}_test_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
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

    test_dataset = SegmentationDataset(
        image_paths, mask_paths,
        transform=img_transform,
        mask_transform=mask_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, classes=cfg.num_classes)

    metrics = {
        "accuracy": torchmetrics.Accuracy(task="binary").to(device),
        "iou": torchmetrics.JaccardIndex(task="binary").to(device),
        "dice": torchmetrics.segmentation.DiceScore(num_classes=cfg.num_classes, average="macro").to(device)
    }

    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, metrics, device)

    print(
        f"Acc: {test_metrics['accuracy']:.4f} | "
        f"IoU: {test_metrics['iou']:.4f} | Dice: {test_metrics['dice']:.4f}"
    )

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_metrics['accuracy'],
        "test_iou": test_metrics['iou'],
        "test_dice": test_metrics['dice'],
    })


if __name__ == "__main__":
    main()
