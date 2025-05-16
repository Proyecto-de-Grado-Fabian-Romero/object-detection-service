import cv2
from ultralytics import YOLO


def load_model(model_path: str = "best.pt") -> YOLO:
    return YOLO(model_path)


def predict(model: YOLO, img, classes=[], conf=0.5):
    if classes:
        return model.predict(img, classes=classes, conf=conf)
    else:
        return model.predict(img, conf=conf)


def predict_and_annotate(
    model: YOLO, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1
):
    results = predict(model, img, classes, conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(
                img,
                (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                (255, 0, 0),
                rectangle_thickness,
            )
            cv2.putText(
                img,
                f"{result.names[int(box.cls[0])]}",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                text_thickness,
            )
    return img, results
