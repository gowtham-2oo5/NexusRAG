import logging
import os
from logging.handlers import RotatingFileHandler

from hackrx_app.utils.constants import LOG_DIR


def configure_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, "hackrx_api.log")

    logger = logging.getLogger("hackrx")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        mode="a",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=False,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("✅ Logger initialized: Console + File")
    logger.info(f"🧪 Logging test: File path is {log_file_path}")

    return logger


