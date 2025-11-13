"""Script utilitaire pour lancer le serveur FastAPI."""
from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import uvicorn


def main() -> None:
    """Démarre l'application FastAPI avec rechargement automatique."""

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
