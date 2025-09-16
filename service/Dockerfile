FROM python:3.12.3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install openslide-python

COPY client.py .
COPY service/app_api.py ./service/
COPY training/models ./training/models
COPY configs ./configs

CMD ["python3", "service/app_api.py"]



