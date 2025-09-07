"""
Split preprocessed images and masks into training and validation sets.

This script prepares file paths for images and their corresponding masks,
verifies that the directories exist, sorts the files to ensure alignment,
and splits them into training and validation sets using scikit-learn's
`train_test_split`.

Workflow:
1. Determine the project root directory relative to this script.
2. Construct full paths to the preprocessed image and mask directories.
3. Check that both directories exist, raise FileNotFoundError if not.
4. List and sort all image and mask filenames to ensure correct pairing.
5. Generate full file paths for images and masks.
6. Split the data into training and validation sets (default 80/20 split).

Dependencies:
    - os
    - sklearn.model_selection.train_test_split

Paths:
    image_dir (str): Directory containing preprocessed image PNG files.
    mask_dir (str): Directory containing corresponding mask PNG files.

Outputs:
    train_imgs (list[str]): Full paths to training images.
    val_imgs (list[str]): Full paths to validation images.
    train_masks (list[str]): Full paths to training masks.
    val_masks (list[str]): Full paths to validation masks.
"""


import os
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

image_dir = os.path.join(PROJECT_ROOT, "data/preprocessed/prep_data/data")
mask_dir  = os.path.join(PROJECT_ROOT, "data/preprocessed/prep_data/markup")

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")
if not os.path.exists(mask_dir):
    raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

images = sorted(os.listdir(image_dir))
masks  = sorted(os.listdir(mask_dir))

image_paths = [os.path.join(image_dir, img) for img in images]
mask_paths  = [os.path.join(mask_dir, msk) for msk in masks]


train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)