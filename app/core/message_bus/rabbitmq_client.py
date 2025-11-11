import json
import logging
import ssl
from typing import Any, Callable, Dict
from urllib.parse import urlparse

import pika


def to_pascal_case(s: str) -> str:
    return "".join(word.capitalize() for word in s.split("_"))


def dict_to_pascal_case(data):
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_key = to_pascal_case(k)
            new_dict[new_key] = dict_to_pascal_case(v)
        return new_dict
    elif isinstance(data, list):
        return [dict_to_pascal_case(i) for i in data]
    else:
        return data


logger = logging.getLogger(__name__)


class RabbitMQClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.channel = None
        self._connect()

    def _parse_connection_string(self):
        """Parsear la connection string de CloudAMQP"""
        connection_string = self.config["RABBITMQ_CONNECTION_STRING"]
        parsed = urlparse(connection_string)

        username = parsed.username
        password = parsed.password
        host = parsed.hostname
        port = parsed.port or 5671
        virtual_host = parsed.path[1:] if parsed.path else "/"

        return {
            "host": host,
            "port": port,
            "virtual_host": virtual_host,
            "credentials": pika.PlainCredentials(username, password),
        }

    def _connect(self):
        """Establecer conexión con CloudAMQP"""
        try:
            connection_params = self._parse_connection_string()
            context = ssl.create_default_context()

            parameters = pika.ConnectionParameters(
                host=connection_params["host"],
                port=connection_params["port"],
                virtual_host=connection_params["virtual_host"],
                credentials=connection_params["credentials"],
                ssl_options=pika.SSLOptions(context),
                heartbeat=600,
                blocked_connection_timeout=300,
            )

            logger.info(
                "🔌 Conectando a RabbitMQ en %s:%s (vhost=%s)...",
                connection_params["host"],
                connection_params["port"],
                connection_params["virtual_host"],
            )

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Declarar exchange y queues
            exchange = self.config["RABBITMQ_EXCHANGE"]
            req_queue = self.config["RABBITMQ_REQUEST_QUEUE"]
            res_queue = self.config["RABBITMQ_RESPONSE_QUEUE"]

            # self.channel.exchange_declare(
            #     exchange=exchange, durable=True, auto_delete=False,
            #     exchange_type='direct')
            self.channel.queue_declare(queue=req_queue, durable=True)
            self.channel.queue_declare(queue=res_queue, durable=True)

            # Aseguramos que las routing keys coincidan
            self.channel.queue_bind(queue=req_queue, exchange=exchange, routing_key=req_queue)
            self.channel.queue_bind(queue=res_queue, exchange=exchange, routing_key=res_queue)

            logger.info(
                "✅ Conectado a RabbitMQ. Exchange='%s', RequestQueue='%s', " "ResponseQueue='%s'",
                exchange,
                req_queue,
                res_queue,
            )

        except Exception as e:
            logger.exception("❌ Error conectando a RabbitMQ: %s", e)
            raise

    def consume_requests(self, callback: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Consumir mensajes de la cola de requests"""
        queue = self.config["RABBITMQ_REQUEST_QUEUE"]

        def on_request(ch, method, props, body):
            corr_id = getattr(props, "correlation_id", None)
            reply_to = getattr(props, "reply_to", None)
            logger.info(
                "📩 Mensaje recibido en '%s' | CorrelationId=%s | ReplyTo=%s",
                queue,
                corr_id,
                reply_to,
                len(body),
            )

            try:
                request = json.loads(body)
                logger.debug("📦 Payload: %s", json.dumps(request, indent=2)[:500])

                response_data = callback(request)

                routing_key = reply_to or self.config["RABBITMQ_RESPONSE_QUEUE"]
                ch.basic_publish(
                    exchange=self.config["RABBITMQ_EXCHANGE"],
                    routing_key=routing_key,
                    properties=pika.BasicProperties(correlation_id=corr_id, delivery_mode=2),
                    body=json.dumps(dict_to_pascal_case(response_data)),
                )

                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(
                    "✅ Procesado OK | CorrelationId=%s -> Respuesta enviada a '%s'",
                    corr_id,
                    routing_key,
                )

            except json.JSONDecodeError as e:
                logger.error("❌ Error decodificando JSON: %s", e)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e:
                logger.exception("❌ Error procesando mensaje: %s", e)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=queue, on_message_callback=on_request, auto_ack=False)

        logger.info("👂 Esperando mensajes en '%s' (Ctrl+C para detener)...", queue)
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.warning("🛑 Consumo detenido manualmente.")
        except Exception as e:
            logger.exception("❌ Error inesperado en el consumidor: %s", e)
        finally:
            self.close()

    def close(self):
        """Cerrar conexión"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("🔒 Conexión RabbitMQ cerrada correctamente")
