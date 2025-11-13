"""Fournit un logger utilisable même sans dépendance externe."""
from __future__ import annotations

import logging
from typing import Any, Callable

try:
    from loguru import logger as _loguru_logger  # type: ignore

    logger = _loguru_logger
except ModuleNotFoundError:
    class _FallbackLogger:
        """Logger minimaliste imitant l'API de base de Loguru."""

        def __init__(self) -> None:
            """Configure un logger standard pour remplacer Loguru."""

            self._logger = logging.getLogger("LegalAssistMA")
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

        def bind(self, **_: Any) -> "_FallbackLogger":
            """Compatibilité avec logger.bind de Loguru."""

            return self

        def opt(self, **_: Any) -> "_FallbackLogger":
            """Compatibilité avec logger.opt de Loguru."""

            return self

        def __getattr__(self, name: str) -> Callable[..., Any]:
            """Redirige les appels vers l'instance logging standard."""

            return getattr(self._logger, name)

    logger = _FallbackLogger()

__all__ = ["logger"]
