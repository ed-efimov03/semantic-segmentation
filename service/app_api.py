from fastapi import FastAPI, UploadFile, File
import segmentation_models_pytorch as smp
import torch
import uvicorn
from omegaconf import OmegaConf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import predict

cfg = OmegaConf.load("../configs/config.yaml")
cfg_unet = OmegaConf.load("../configs/model/unet.yaml")
cfg_deeplab = OmegaConf.load("../configs/model/deeplabv3.yaml")

def load_model(cfg):
    model_name = cfg.defaults[2].model
    if model_name == "unet":
        model = smp.Unet(
            encoder_name=cfg_unet.encoder,
            in_channels=3,
            classes=cfg.num_classes
        )
        model_path = "../training/models/unet_custom.pth"

    elif model_name == "deeplabv3":
        model = smp.DeepLabV3Plus(
            encoder_name=cfg_deeplab.encoder,
            in_channels=3,
            classes=cfg.num_classes
        )
        model_path = "../training/models/deeplabv3+_custom.pth"

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
    """Загружаем модель один раз при старте приложения"""
    global model
    model = load_model(cfg)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    return await predict(model, file) 


@app.get("/health")
def health():
    return {"status": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)

