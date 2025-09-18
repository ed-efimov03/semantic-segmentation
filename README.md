# 🧪🔬🤖 Tissue Segmentation on Whole Slide Images (WSI) and FastAPI Inference Service

## 📌 Project Description
The goal of this project is to train segmentation models for detecting tissue regions in Whole Slide Images (WSI) and to develop a FastAPI-based inference service for the trained model.
A complete pipeline is implemented: from data preparation and model training to service deployment in Docker and writing a client script.

Note: Steps 2 and 3 are optional (only needed if you want to retrain the models).
---

## 🏗️ Project Structure
```bash
project_root/
├── training/                # Code for data preparation and training
├── service/                 # FastAPI + Dockerfile
├── client.py                # Client script
├── configs/                 # Configs
├── README.md                # Setup instructions
└── requirements.txt         # Dependencies
```

---

## ⚙️ Environment Setup

### 1. Installing dependencies
It is recommended to use [UV](https://docs.astral.sh/uv/) or `conda`.  
Example installation of `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Example usage of `uv`:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

It is also recommended to install QuPath for WSI viewing and annotation.


### 2. Data Preparation
Download the datasets (provided in a zip file):
- Main dataset (WSI + GeoJSON)
- Test dataset: [Histo-Seg](https://data.mendeley.com/datasets/vccj8mp2cg/1)

Main dataset:
- Transfer the downloaded files in `training/data/raw`
- Extract files using the `extract_zip.py` utility
- Go to `../preprocessed`
- Process data using the `data_preprocess.py` utility

Test dataset:
- Transfer the downloaded files in `training/data/test`
- Extract files using the `extract_histo_test.py` utility
- Process data using the `preprocess.py` utility

### 3.Training and Testing
- Navigate to `training`
- Run `train.py` to train the model

Example training run:
```bash
uv run train.py model="deeplabv3"
```

- Run `test.py` to evaluate the model

Example test run:
```bash
uv run test.py model="deeplabv3"
```


### 4. Running the FastAPI Service
Instructions if you performed steps 2 and 3:
- Navigate to the project root
- Build the Docker image:
```bash
docker build -f service/Dockerfile -t semantic-segmentation:latest .
```
- Run the Docker container:
```bash
docker run -d --name semantic-segmentation -p <your port number>:5000 semantic-segmentation:latest
```
- Add port `your port number`
- Use model




Instructions if you did NOT perform steps 2 and 3:
- Go to `training`
- Create a `models` folder and navigate into it
- Download the pretrained models: 
[unet](https://drive.google.com/file/d/1c_ZwHinynT-qnC12o1-leySzNcZp1Bxa/view?usp=drive_link), 
[deeplabv3](https://drive.google.com/file/d/15Bn2ASY_UYJjsZeivyXJmVZvsX2Ch4Bp/view?usp=drive_link)

Then follow the same steps as in the previous case:
- Navigate to the project root
- Build the Docker image:
```bash
docker build -f service/Dockerfile -t semantic-segmentation:latest .
```
- Run the Docker container:
```bash
docker run -d --name semantic-segmentation -p <your port number>:5000 semantic-segmentation:latest
```
- Add port `your port number`
- Use model