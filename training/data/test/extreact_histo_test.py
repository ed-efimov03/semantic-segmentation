"""
Dataset Extraction Script.

This script extracts the contents of a ZIP archive into the current working directory. 
It is intended for preparing the "Histo-Seg H&E Whole Slide Image Segmentation Dataset" 
for further preprocessing and training.

Libraries:
    zipfile: Provides tools for reading and extracting ZIP archives.

Usage:
    - Ensure the dataset ZIP file is located in the same directory as this script.
    - Run the script to extract all contents into the current folder.

Input:
    - "Histo-Seg H&E Whole Slide Image Segmentation Dataset.zip": 
      The compressed dataset file.

Output:
    - Extracted dataset files and directories in the current working directory.
"""


import zipfile

with zipfile.ZipFile("Histo-Seg H&E Whole Slide Image Segmentation Dataset.zip", "r") as zf:
    zf.extractall("")