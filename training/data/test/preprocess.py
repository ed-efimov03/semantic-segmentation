"""
Histopathology Dataset Preprocessing Script.

This script prepares histopathology images and their corresponding masks 
for segmentation tasks. It performs resizing, normalization of masks to binary 
format, and saves the processed files into new output directories.

Processing steps:
    - Masks:
        * Read in grayscale.
        * Binarize (all non-zero pixels set to 1).
        * Resize to target size (nearest neighbor to preserve labels).
        * Save as PNG files.
    - Images:
        * Read in color (BGR).
        * Resize to target size (area interpolation for smoother scaling).
        * Save as PNG files.

Input:
    - Masks: JPG images in `input_folder_masks`.
    - Images: JPG images in `input_folder_img`.

Output:
    - Processed PNG masks saved in `output_folder_masks`.
    - Processed PNG images saved in `output_folder_img`.

Constants:
    input_folder_masks (str): Path to original mask images.
    output_folder_masks (str): Path to save processed mask images.
    input_folder_img (str): Path to original histopathology images.
    output_folder_img (str): Path to save processed histopathology images.
    target_size (tuple): Desired (width, height) for resizing.

Libraries:
    cv2: For image reading, resizing, and writing.
    numpy: For binary mask conversion.
    glob: For batch file matching.
    os: For path operations and directory management.

Notes:
    - Masks are stored as binary images (0 and 1).
    - Both masks and images are resized to 512x512 pixels.
"""


import cv2
import numpy as np
import glob
import os

input_folder_masks = "Histo-Seg-H&E-Whole-Slide-Image-Segmentation-Dataset/Histo-Seg/Masks"
output_folder_masks = "Histo-Seg-H&E-Whole-Slide-Image-Segmentation-Dataset/Histo-Seg/Masks-prep"

input_folder_img = "Histo-Seg-H&E-Whole-Slide-Image-Segmentation-Dataset/Histo-Seg/Images"
output_folder_img = "Histo-Seg-H&E-Whole-Slide-Image-Segmentation-Dataset/Histo-Seg/Images-prep"

os.makedirs(output_folder_masks, exist_ok=True)
os.makedirs(output_folder_img, exist_ok=True)



target_size = (512, 512)


for path in glob.glob(os.path.join(input_folder_masks, "*.jpg")):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    

    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)


    resized_mask = cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST)


    filename = os.path.splitext(os.path.basename(path))[0] + ".png"
    out_path = os.path.join(output_folder_masks, filename)
    cv2.imwrite(out_path, resized_mask)


for path in glob.glob(os.path.join(input_folder_img, "*.jpg")):
    img = cv2.imread(path)
    
  
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


    filename = os.path.splitext(os.path.basename(path))[0] + ".png"
    out_path = os.path.join(output_folder_img, filename)
    cv2.imwrite(out_path, resized_img)



