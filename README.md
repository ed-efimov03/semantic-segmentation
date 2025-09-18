# 🧪🔬🤖 Сегментация тканей на Whole Slide Images (WSI) и FastAPI сервис инференса

## 📌 Описание проекта
Цель проекта — обучение моделей сегментации областей, содержащих ткань, на Whole Slide Images (WSI), а также разработка FastAPI-сервиса для инференса обученной модели.  
Реализован полный pipeline: от подготовки данных и обучения моделей до развёртывания сервиса в Docker и написания клиентского скрипта.

Замечание: пункты 2 и 3 не обязательны (их нужно делать, если вы хотите заново обучить модели)
---

## 🏗️ Структура проекта
project_root/
├── training/                # Код для подготовки и обучения
├── service/                 # FastAPI + Dockerfile
├── client.py                # Клиентский скрипт
├── configs/                 # Конфиги
├── README.md                # Инструкции по запуску
└── requirements.txt         # Зависимости

---

## ⚙️ Установка окружения

### 1. Установка зависимостей
Рекомендуется использовать [UV](https://docs.astral.sh/uv/) или `conda`.  
Пример уставновки `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Пример использования `uv`:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Также рекомендуется уставновить программу QuPath для просмотра WSI и разметки


### 2. Подготовка данных
Скачайте данные (будут лежать в zip файле):
- Основной датасет (WSI + GeoJSON) 
- Тестовый датасет: [Histo-Seg](https://data.mendeley.com/datasets/vccj8mp2cg/1)

Основной датасет:
- Положить скачанные данные в training/data/raw
- Извлечь файлы с помощью утилиты extract_zip.py
- Перейти в ../preprocessed
- Обработать данные с помощью утилиты data_preprocess.py

Тестовый датасет:
- Положить скачанные данные в training/data/test
- Извлечь файлы с помощью утилиты extract_histo_test.py
- Обработать данные с помощью утилиты preprocess.py

### 3.Обучение и тест
- Перейти в training
- Запустить утилиту train.py для обучения

Пример запуска обучения:
```bash
uv run train.py model="deeplabv3"
```
- Запустить утилиту test.py для теста

Пример запуска теста:
```bash
uv run test.py model="deeplabv3"
```


### 4. Запуск Fast_API сервиса
Инструкция по запуску сервиса, если вы выполняли пункты 2 и 3:
- Перейти в корневую папку проекта
- Собртать образ docker 
```bash
docker build -f service/Dockerfile -t semantic-segmentation:latest .
```
- Запустить docker контейнер
```bash
docker run -d --name semantic-segmentation -p <your port number>:5000 semantic-segmentation:latest
```
- Add port <your port number>



Инструкция по запуску сервиса, если вы НЕ выполняли пункты 2 и 3:
- Перейти в training
- Создать папку models и перейти в неё
- Загрузить обученную [unet](https://drive.google.com/file/d/1c_ZwHinynT-qnC12o1-leySzNcZp1Bxa/view?usp=drive_link) и [deeplabv3](https://drive.google.com/file/d/15Bn2ASY_UYJjsZeivyXJmVZvsX2Ch4Bp/view?usp=drive_link)
Дальше также как и в другом варианте
- Перейти в корневую папку проекта
- Собртать образ docker 
```bash
docker build -f service/Dockerfile -t semantic-segmentation:latest .
```
- Запустить docker контейнер
```bash
docker run -d --name semantic-segmentation -p <your port number>:5000 semantic-segmentation:latest
```
- Add port <your port number>