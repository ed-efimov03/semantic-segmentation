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

cfg_img_size = OmegaConf.load("../configs/dataset/default.yaml")

def transform(size):
    img_transform = transforms.Compose([
        transforms.Resize(tuple(size)),
        transforms.ToTensor()
    ])
    return img_transform


async def predict(model, file):
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
        mask_bin = (mask > 127).astype("uint8")
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def scale_contours(contours):
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