import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    """
    This dataset class loads images and their corresponding segmentation masks,
    applies optional transformations, and returns them as tensors suitable for
    training or evaluation in PyTorch models.

    Attributes:
        image_paths (list[str]): List of file paths to input images.
        mask_paths (list[str]): List of file paths to corresponding masks.
        transform (callable, optional): Transformations to apply to images.
        mask_transform (callable, optional): Transformations to apply to masks.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the (image, mask) pair at index `idx`.

    Notes:
        - Images are converted to RGB format.
        - Masks are converted to single-channel (grayscale) format.
        - If no mask_transform is provided, masks are converted to long tensors.
    """

    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return img, mask
