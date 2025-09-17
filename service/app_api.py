"""
FastAPI Server for Whole-Slide Image Segmentation

This module provides a REST API for performing whole-slide image (WSI) segmentation 
using pretrained PyTorch models. It supports UNet and DeepLabV3+ models via the 
`segmentation_models_pytorch` library and serves predictions along with health checks.

Endpoints:
    - POST /predict: Accepts a slide image and returns a ZIP file containing the 
      predicted mask and corresponding GeoJSON contours.
    - GET /health: Returns the health status of the server.

Startup:
    - Loads the segmentation model once when the server starts.

Dependencies:
    - fastapi
    - segmentation_models_pytorch
    - torch
    - uvicorn
    - omegaconf
"""


from fastapi import FastAPI, UploadFile, File
import segmentation_models_pytorch as smp
import torch
import uvicorn
from omegaconf import OmegaConf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import predict

cfg = OmegaConf.load("configs/config.yaml")
cfg_unet = OmegaConf.load("configs/model/unet.yaml")
cfg_deeplab = OmegaConf.load("configs/model/deeplabv3.yaml")

def load_model(cfg):
    """
    Loads the segmentation model specified in the configuration.

    Supports UNet and DeepLabV3+ architectures.

    Args:
        cfg (omegaconf.dictconfig.DictConfig): Configuration specifying model type,
            encoder, number of classes, and device.

    Returns:
        torch.nn.Module: The loaded PyTorch model set to evaluation mode.

    Raises:
        ValueError: If an unknown model type is specified in the configuration.
    """

    model_name = cfg.defaults[2].model
    if model_name == "unet":
        model = smp.Unet(
            encoder_name=cfg_unet.encoder,
            in_channels=3,
            classes=cfg.num_classes
        )
        model_path = "training/models/unet_custom.pth"

    elif model_name == "deeplabv3":
        model = smp.DeepLabV3Plus(
            encoder_name=cfg_deeplab.encoder,
            in_channels=3,
            classes=cfg.num_classes
        )
        model_path = "training/models/deeplabv3+_custom.pth"

    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(cfg.device if torch.cuda.is_available() else "cpu")
    model.eval()

    return model


app = FastAPI()
 
@app.on_event("startup")
def startup_event():
    """
    Startup event handler for FastAPI.

    Loads the segmentation model once and stores it in a global variable.
    """
    global model
    model = load_model(cfg)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to perform segmentation on an uploaded slide image.

    Args:
        file (UploadFile): The uploaded whole-slide image.

    Returns:
        fastapi.responses.FileResponse: ZIP file containing:
            - Mask PNG of the segmentation.
            - GeoJSON file of the mask contours.
    """
    return await predict(model, file) 


@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns:
        dict: Health status of the server.
    """

    return {"status": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

