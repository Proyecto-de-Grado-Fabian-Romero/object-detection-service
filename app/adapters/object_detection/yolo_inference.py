import logging
import os
from typing import Optional

import requests
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

# Carga las variables del archivo .env si estás corriendo localmente
load_dotenv()

logger = logging.getLogger(__name__)

# Singleton: model loaded once per process
_model: Optional[YOLO] = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_use_half: bool = torch.cuda.is_available()


def load_model() -> YOLO:
    """
    Carga el modelo YOLO desde la ruta local indicada en YOLO_MODEL_PATH.
    Si el modelo no existe localmente, lo descarga desde YOLO_MODEL_URL.
    El modelo se mantiene en memoria como singleton para evitar recargas.
    """
    global _model
    if _model is not None:
        return _model

    model_path = os.getenv("YOLO_MODEL_PATH", "models/yolo11x.pt")
    model_url = os.getenv("YOLO_MODEL_URL")

    # Si el modelo no existe, descargarlo automáticamente
    if not os.path.exists(model_path):
        if not model_url:
            raise ValueError(
                f"El modelo no existe en {model_path} y no se especificó YOLO_MODEL_URL en el .env"
            )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logger.info(f"Descargando modelo YOLO desde {model_url} ...")

        response = requests.get(model_url, stream=True, timeout=300)
        response.raise_for_status()

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Modelo guardado en {model_path}")

    logger.info(f"Cargando modelo YOLO en dispositivo '{_device}' (half={_use_half}) ...")
    _model = YOLO(model_path)
    _model.to(_device)
    logger.info("Modelo YOLO cargado exitosamente")
    return _model


def predict(model: YOLO, img, classes=[], conf=0.5):
    kwargs = dict(
        conf=conf,
        imgsz=640,
        device=_device,
        half=_use_half,
        verbose=False,
    )
    if classes:
        kwargs["classes"] = classes
    return model.predict(img, **kwargs)


def predict_and_annotate(model: YOLO, img, classes=[], conf=0.5, rectangle_thickness=2):
    results = predict(model, img, classes, conf)
    return img, results
