"""
Whole-slide Image Segmentation and Contour Extraction

This module provides functionality to process whole-slide images (WSIs) using a 
PyTorch segmentation model. It extracts a thumbnail of the slide, performs model 
inference, generates a segmentation mask, extracts contours, scales them to the 
original slide dimensions, converts them to GeoJSON, and packages the results into a ZIP file.

Dependencies:
    - torch
    - omegaconf
    - PIL
    - openslide
    - io
    - torchvision.transforms
    - tempfile
    - cv2
    - fastapi.responses
    - json
    - zipfile

Constants:
    Image.MAX_IMAGE_PIXELS: Allows loading of very large images without PIL warnings.
    cfg_img_size: Image size configuration loaded from `configs/dataset/default.yaml`.

Functions:
    transform(size)
    predict(model, file)
    mask_to_contours(mask)
    scale_contours(contours)
    contours_to_geojson(contours)
"""


import torch
from omegaconf import OmegaConf
from PIL import Image   
import openslide
import io
import torchvision.transforms as transforms
import tempfile
import cv2
from fastapi.responses import FileResponse
import json
import zipfile

Image.MAX_IMAGE_PIXELS = None 

cfg_img_size = OmegaConf.load("configs/dataset/default.yaml")

def transform(size):
    """
    Creates a PyTorch image transformation pipeline.

    The pipeline resizes the image to the specified size and converts it to a tensor.

    Args:
        size (tuple): Desired image size (width, height).

    Returns:
        torchvision.transforms.Compose: Composed transformation.
    """

    img_transform = transforms.Compose([
        transforms.Resize(tuple(size)),
        transforms.ToTensor()
    ])
    return img_transform


async def predict(model, file):
    """
    Predicts a segmentation mask from a whole-slide image and generates corresponding 
    contours in GeoJSON format, then returns both as a ZIP file.

    This function performs the following steps:
        1. Reads the uploaded slide image file.
        2. Creates a temporary file to store the uploaded slide.
        3. Loads the slide using OpenSlide and extracts a thumbnail.
        4. Applies transformations to resize and convert the image to a tensor.
        5. Performs model inference to generate a segmentation mask.
        6. Converts the mask to a binary or multi-class mask as needed.
        7. Saves the mask as a PNG file.
        8. Extracts contours from the mask using OpenCV.
        9. Scales contours to the original slide dimensions.
        10. Converts contours to GeoJSON format.
        11. Saves both the mask PNG and GeoJSON file to a temporary ZIP file.
        12. Returns the ZIP file as a FastAPI `FileResponse`.

    Args:
        model (torch.nn.Module): The PyTorch model used for segmentation inference.
        file (UploadFile): The uploaded slide file to process.

    Returns:
        fastapi.responses.FileResponse: A response containing a ZIP file with:
            - `{original_filename}_mask.png`: The predicted segmentation mask.
            - `{original_filename}_mask.geojson`: The corresponding contours in GeoJSON format.

    Raises:
        ValueError: If the generated mask does not have a valid 2D shape.
    """

    content = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    slide = openslide.OpenSlide(tmp_path)

    img_size = slide.dimensions

    thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
    image = thumbnail.convert("RGB")

    img_transform = transform(cfg_img_size.img_size)
    img_tensor = img_transform(image).unsqueeze(0) 
    img_tensor = img_tensor.to(next(model.parameters()).device)


    with torch.no_grad():
        output = model(img_tensor)  

    num_classes = output.shape[1]
    if num_classes == 1:
        mask = (output.squeeze(0).squeeze(0) > 0.5).cpu().numpy().astype("uint8") * 255  
    else:
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype("uint8")  
        mask = (mask / mask.max() * 255).astype("uint8") if mask.max() > 0 else mask


    mask_img = Image.fromarray(mask)
    tmp_mask = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    mask_img.save(tmp_mask.name)

    if len(mask.shape) == 3:
        if mask.shape[0] == 1:
            mask = mask[0]  
        else:
            mask = mask.squeeze()
    
    if len(mask.shape) != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    

    mask_img = Image.fromarray(mask)


    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)



    def mask_to_contours(mask):
        """
        Converts a binary mask to contours using OpenCV.

        Args:
            mask (np.ndarray): 2D mask array with values 0 or 255.

        Returns:
            list: List of contours detected in the mask.
        """

        mask_bin = (mask > 127).astype("uint8")
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def scale_contours(contours):
        """
        Scales contours from thumbnail dimensions to original slide dimensions.

        Args:
            contours (list): List of contours in thumbnail coordinates.

        Returns:
            list: List of scaled contours in original slide coordinates.
        """

        slide_w, slide_h = slide.level_dimensions[0]
        thumb_w, thumb_h = cfg_img_size.img_size
        scale_x = slide_w / thumb_w
        scale_y = slide_h / thumb_h
        scaled_contours = []
        for contour in contours:
            scaled = contour.astype("float32")
            scaled[:, 0, 0] *= scale_x
            scaled[:, 0, 1] *= scale_y
            scaled_contours.append(scaled.tolist())
        return scaled_contours

    def contours_to_geojson(contours):
        """
        Converts contours into a GeoJSON FeatureCollection.

        Args:
            contours (list): List of scaled contours.

        Returns:
            dict: GeoJSON FeatureCollection representing the contours.
        """
        
        features = []
        for contour in contours:
            if len(contour) < 3:
                continue
            coords = [[int(point[0][0]), int(point[0][1])] for point in contour]
            coords.append(coords[0]) 
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {}
            })
        return {"type": "FeatureCollection", "features": features}

    contours = mask_to_contours(mask)
    scaled_contours = scale_contours(contours)
    geojson_data = contours_to_geojson(scaled_contours)



    tmp_geojson = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
    with open(tmp_geojson.name, "w") as f:
        json.dump(geojson_data, f)

    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(tmp_zip.name, "w") as zipf:
        zipf.write(tmp_mask.name, arcname=f"{file.filename.split('.')[0]}_mask.png")
        zipf.write(tmp_geojson.name, arcname=f"{file.filename.split('.')[0]}_mask.geojson")



    return FileResponse(
        tmp_zip.name,
        media_type="application/zip",
        filename=f"{file.filename.split('.')[0]}_results.zip"
    )