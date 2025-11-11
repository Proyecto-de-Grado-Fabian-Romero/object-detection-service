import os
import threading
import logging
from flask import Flask
from flasgger import Swagger

from app.routes.detect_routes import detect_blueprint
from app.routes.preprocess_routes import preprocess_blueprint
from app.routes.process_routes import process_blueprint
from app.routes.detection_consumer import start_detection_consumer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)

    app.config["RABBITMQ_CONNECTION_STRING"] = os.getenv(
        "RABBITMQ_CONNECTION_STRING",
        "amqps://uzzvrcdk:w8ZOzXjKJ60pSP3Uk-7V_Nmb1UJqRNZT" "@gorilla.lmq.cloudamqp.com/uzzvrcdk",
    )
    app.config["RABBITMQ_EXCHANGE"] = os.getenv("RABBITMQ_EXCHANGE", "spacio.direct")
    app.config["RABBITMQ_REQUEST_QUEUE"] = "detection_requests"
    app.config["RABBITMQ_RESPONSE_QUEUE"] = "detection_responses"
    app.config["START_RABBITMQ_CONSUMER"] = (
        os.getenv("START_RABBITMQ_CONSUMER", "true").lower() == "true"
    )

    app.config["SWAGGER"] = {
        "title": "Object Detection API",
        "uiversion": 3,
    }
    Swagger(app)

    app.register_blueprint(preprocess_blueprint, url_prefix="/preprocess")
    app.register_blueprint(detect_blueprint, url_prefix="/detect")
    app.register_blueprint(process_blueprint, url_prefix="/process")

    if app.config["START_RABBITMQ_CONSUMER"]:
        try:
            consumer_thread = threading.Thread(
                target=start_detection_consumer, args=(app.config,), daemon=True
            )
            consumer_thread.start()
            logger.info("✅ RabbitMQ consumer started in background thread")
            logger.info(f"📡 Exchange: {app.config['RABBITMQ_EXCHANGE']}")
            logger.info(f"📥 Queue: {app.config['RABBITMQ_REQUEST_QUEUE']}")
        except Exception as e:
            logger.error(f"❌ Failed to start RabbitMQ consumer: {e}")

    @app.route("/health")
    def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "rabbitmq_configured": bool(app.config["RABBITMQ_CONNECTION_STRING"]),
            "exchange": app.config["RABBITMQ_EXCHANGE"],
            "consumer_enabled": app.config["START_RABBITMQ_CONSUMER"],
        }

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("🚀 Starting Object Detection Service on port 5151")
    app.run(port=5151, debug=True)
