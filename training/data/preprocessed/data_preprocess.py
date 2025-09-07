"""
Generate thumbnails and corresponding masks from WSI (Whole Slide Images).

This script processes all .tif images in the `input_dir`, creates thumbnails,
and generates binary masks based on corresponding GeoJSON annotations. 
The outputs are saved in separate directories for images and masks.

Workflow:
1. Traverse the `input_dir` recursively to find all .tif files.
2. For each .tif file:
   - Load the WSI using OpenSlide.
   - Generate a thumbnail of the smallest resolution level.
   - Save the thumbnail as a PNG in `path_data`.
   - Load the corresponding GeoJSON annotation.
   - Create a binary mask at the thumbnail resolution.
   - Scale polygon coordinates from WSI size to thumbnail size.
   - Draw the polygons onto the mask.
   - Save the mask as a PNG in `path_markup`.

Dependencies:
    - openslide
    - numpy
    - Pillow (PIL)
    - os
    - json

Paths:
    input_dir (str): Path to raw WSI files.
    path_data (str): Directory to save generated PNG thumbnails.
    path_markup (str): Directory to save generated PNG masks.

Notes:
    - Masks are binary images (mode "L") with 1 inside annotated regions and 0 outside.
    - The script expects each WSI to have a corresponding GeoJSON file with the same base filename.
"""

import openslide
import json
import numpy as np
from PIL import Image, ImageDraw
import os

path_data = "prep_data/data"
path_markup = "prep_data/markup"
os.makedirs(path_data, exist_ok=True)
os.makedirs(path_markup, exist_ok=True)


input_dir = "../raw/internship_subset/data"

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        filepath = os.path.join(root, filename)

        filename_geojson = filename.replace(".tif", ".geojson")
        filename_png_image = filename.replace(".tif", ".png")
        filename_png_mask = filename_png_image.replace("DHMC", "mask")


        slide = openslide.OpenSlide(filepath) 
        thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
        thumbnail.save(f"{path_data}/{filename_png_image}")


        wsi_width, wsi_height = slide.level_dimensions[0]
        thumb_width, thumb_height = slide.level_dimensions[-1]

        scale_x = thumb_width / wsi_width
        scale_y = thumb_height / wsi_height



        with open(f"../raw/internship_subset/markup/{filename_geojson}") as f:
            geojson = json.load(f)

        mask = Image.new("L", (thumb_width, thumb_height), 0)
        draw = ImageDraw.Draw(mask)

        for feature in geojson["features"]:
            coords_group = feature["geometry"]["coordinates"]
           
            if feature["geometry"]["type"] == "Polygon":
                rings = [coords_group]
           
            elif feature["geometry"]["type"] == "MultiPolygon":
                rings = coords_group
            else:
                continue 

            for ring in rings:
                coords = ring[0] 
                scaled_coords = [(x * scale_x, y * scale_y) for x, y in coords]
                draw.polygon(scaled_coords, outline=0, fill=1)

        mask.save(f"{path_markup}/{filename_png_mask}")

