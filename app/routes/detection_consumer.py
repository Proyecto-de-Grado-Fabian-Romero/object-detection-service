import logging
import os
from io import BytesIO

import requests
from PIL import Image
from werkzeug.utils import secure_filename

from app.core.message_bus.rabbitmq_client import RabbitMQClient
from app.usecases.messages.detection_messages import (
    DetectionRequest,
    DetectionResponse,
)
from app.usecases.postprocess_detections import (
    postprocess_detections_with_tracking,
)
from app.usecases.preprocess_equirect import preprocess_image
from app.usecases.run_object_detection import run_detection_on_folder

UPLOAD_FOLDER = "temp_uploads"
PREPROCESS_OUTPUT = "output_views"

logger = logging.getLogger(__name__)


def process_detection_request(request_data: dict) -> dict:
    """Process detection request from RabbitMQ"""
    request = DetectionRequest(**request_data)
    logger.info("✅ called")

    try:
        aggregated_objects = {}

        for url in request.image_urls:
            response = requests.get(url)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            filename = secure_filename(url.split("/")[-1])
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            image.save(filepath)

            # Process image
            folder = preprocess_image(filepath, PREPROCESS_OUTPUT)
            detections = run_detection_on_folder(folder)
            objects_count = postprocess_detections_with_tracking(detections, filepath)

            # Count objects
            for class_id, data in objects_count.items():
                if class_id not in aggregated_objects:
                    aggregated_objects[class_id] = 0
                aggregated_objects[class_id] += data["count"]

            # Cleanup
            os.remove(filepath)
            for preprocessed_img in os.listdir(folder):
                os.remove(os.path.join(folder, preprocessed_img))
            os.rmdir(folder)

        response = DetectionResponse(
            request_id=request.request_id,
            detected_objects=aggregated_objects,
            success=True,
        )

    except Exception as e:
        response = DetectionResponse(
            request_id=request.request_id,
            detected_objects={},
            success=False,
            error=str(e),
        )

    return response.dict()


def start_detection_consumer(config):
    """Start the RabbitMQ consumer for detection requests"""
    try:
        rabbitmq_client = RabbitMQClient(config)
        rabbitmq_client.consume_requests(process_detection_request)
    except Exception as e:
        print(f"Failed to start RabbitMQ consumer: {e}")
