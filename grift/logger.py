import json
import logging
from datetime import datetime, UTC
from typing import Any, Dict, Optional, Union, Protocol, Type

# Define the protocol and base handler types first
class CloudLoggingClient(Protocol):
    def get_default_handler(self) -> logging.Handler: ...

class StructuredLogHandler(logging.Handler):
    """Base StructuredLogHandler that will be used if cloud logging is not available."""
    def emit(self, record: logging.LogRecord) -> None:
        pass

# Type definitions and imports
cloud_logging = None
Handler = logging.Handler
Client = CloudLoggingClient

try:
    from google.cloud import logging as cloud_logging
    from google.cloud.logging.handlers import StructuredLogHandler as CloudStructuredLogHandler
    Handler = Union[logging.Handler, CloudStructuredLogHandler]
    StructuredLogHandler = CloudStructuredLogHandler
    Client = cloud_logging.Client
except ImportError:
    pass  # Use the default definitions from above

from grift.config import settings


class StructuredLogger:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

        handler: Handler
        if cloud_logging and settings.environment != "development":
            # Set up Google Cloud Logging if available and not in development
            client: Client = cloud_logging.Client()  # type: ignore
            handler = client.get_default_handler()
        else:
            # Local development logging
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)

    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "service": self.name,
            "environment": settings.environment,
            "message": message
        }
        if extra:
            log_data.update(extra)
        return json.dumps(log_data)

    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message, kwargs))

    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message, kwargs))

    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message(message, kwargs))

    def error(self, message: str, **kwargs):
        self.logger.error(self._format_message(message, kwargs))

    def critical(self, message: str, **kwargs):
        self.logger.critical(self._format_message(message, kwargs))


# Example usage:
# logger = StructuredLogger("tick-fetcher")
# logger.info("Processing tick", instrument="EUR_USD", price=1.1234)
