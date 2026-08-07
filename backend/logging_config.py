from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_persistent_logging(
    logger: logging.Logger,
    *,
    backup_count: int,
    level_name: str,
    max_bytes: int,
    path: str,
) -> Path:
    """Attach one rotating UTF-8 file handler and return its resolved path."""
    level = getattr(logging, level_name.upper())
    log_path = Path(path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(level)

    for handler in logger.handlers:
        if getattr(handler, "p_gpt_persistent_log", False):
            handler.setLevel(level)
            return log_path

    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.p_gpt_persistent_log = True  # type: ignore[attr-defined]
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    return log_path
