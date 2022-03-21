import logging
from logging import handlers
from pathlib import Path

from rich.logging import RichHandler


def create_logger(log_dir):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = RichHandler(markup=True)
    console_handler.setLevel(logging.DEBUG)
    info_handler = logging.handlers.RotatingFileHandler(
        filename=Path(log_dir, "info.log"),
        maxBytes=10485760,  # 1 MB
        backupCount=10,
    )
    info_handler.setLevel(logging.INFO)
    error_handler = logging.handlers.RotatingFileHandler(
        filename=Path(log_dir, "error.log"),
        maxBytes=10485760,  # 1 MB
        backupCount=10,
    )
    error_handler.setLevel(logging.ERROR)

    # Create formatters
    minimal_formatter = logging.Formatter(fmt="%(message)s")
    detailed_formatter = logging.Formatter(
        fmt="%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
    )

    # Hook it all up
    console_handler.setFormatter(fmt=minimal_formatter)
    info_handler.setFormatter(fmt=detailed_formatter)
    error_handler.setFormatter(fmt=detailed_formatter)
    logger.addHandler(hdlr=console_handler)
    logger.addHandler(hdlr=info_handler)
    logger.addHandler(hdlr=error_handler)

    return logger
