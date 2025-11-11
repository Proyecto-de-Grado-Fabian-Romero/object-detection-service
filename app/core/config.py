import os
from urllib.parse import urlparse


class Config:
    """Configuración de la aplicación"""

    RABBITMQ_CONNECTION_STRING = os.getenv(
        "RABBITMQ_CONNECTION_STRING",
        "amqps://uzzvrcdk:w8ZOzXJjKJ60pSP3Uk-7V_Nmb1UJqRNZT" "@gorilla.lmq.cloudamqp.com/uzzvrcdk",
    )
    RABBITMQ_EXCHANGE = os.getenv("RABBITMQ_EXCHANGE", "spacio.direct")

    # Queues names
    RABBITMQ_REQUEST_QUEUE = "detection_requests"
    RABBITMQ_RESPONSE_QUEUE = "detection_responses"

    # Consumer settings
    START_RABBITMQ_CONSUMER = os.getenv("START_RABBITMQ_CONSUMER", "true").lower() == "true"

    @property
    def RABBITMQ_VHOST(self):
        """Extraer vhost de la connection string"""
        parsed = urlparse(self.RABBITMQ_CONNECTION_STRING)
        return parsed.path[1:] if parsed.path else "/"


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
