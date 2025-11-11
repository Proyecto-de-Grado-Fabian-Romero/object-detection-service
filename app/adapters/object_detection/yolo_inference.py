import os
import requests
from dotenv import load_dotenv
from ultralytics import YOLO

# Carga las variables del archivo .env si estás corriendo localmente
load_dotenv()


def load_model() -> YOLO:
    """
    Carga el modelo YOLO desde la ruta local indicada en YOLO_MODEL_PATH.
    Si el modelo no existe localmente, lo descarga desde YOLO_MODEL_URL.
    """
    model_path = os.getenv("YOLO_MODEL_PATH", "models/yolo11x.pt")
    model_url = os.getenv("YOLO_MODEL_URL")

    # Si el modelo no existe, descargarlo automáticamente
    if not os.path.exists(model_path):
        if not model_url:
            raise ValueError(
                f"El modelo no existe en {model_path} y no se especificó YOLO_MODEL_URL"
                "en el .env"
            )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"📥 Descargando modelo desde {model_url} ...")

        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ Modelo descargado en {model_path}")

    # Cargar el modelo YOLO
    print(f"🚀 Cargando modelo desde {model_path}")
    return YOLO(model_path)


def predict(model: YOLO, img, classes=[], conf=0.5):
    if classes:
        return model.predict(img, classes=classes, conf=conf, imgsz=512)
    else:
        return model.predict(img, conf=conf, imgsz=512)


def predict_and_annotate(
    model: YOLO, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1
):
    results = predict(model, img, classes, conf)
    # for result in results:
    #     for box in result.boxes:
    #         cv2.rectangle(
    #             img,
    #             (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
    #             (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
    #             (255, 0, 0),
    #             rectangle_thickness,
    #         )
    #         cv2.putText(
    #             img,
    #             f"{result.names[int(box.cls[0])]}",
    #             (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
    #             cv2.FONT_HERSHEY_PLAIN,
    #             1,
    #             (255, 0, 0),
    #             text_thickness,
    #         )
    return img, results
