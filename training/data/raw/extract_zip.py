"""
Extract the contents of the internship_subset.zip archive.

This script opens the ZIP file located at "internship_subset.zip"
and extracts all its contents into the "" directory.

Workflow:
1. Open the ZIP file in read mode.
2. Extract all files and folders into the target directory.

Notes:
- If the archive does not exist, a FileNotFoundError may be raised.
- The internal folder structure of the archive is preserved.

Dependencies:
- zipfile (Python standard library)
"""

import zipfile

with zipfile.ZipFile("internship_subset.zip", "r") as zf:
    zf.extractall("")