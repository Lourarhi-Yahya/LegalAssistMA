"""Téléchargement des modèles nécessaires au projet."""
from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger

from config import settings


def download_whisper(model_size: str) -> None:
    """Télécharge le modèle Whisper sélectionné via l'API openai-whisper."""

    logger.info("Téléchargement du modèle Whisper %s", model_size)
    subprocess.run(["whisper", "--model", model_size, "--list-models"], check=False)


def download_spacy(model_name: str) -> None:
    """Télécharge le modèle spaCy requis."""

    logger.info("Téléchargement du modèle spaCy %s", model_name)
    subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)


def prepare_directories() -> None:
    """Crée les répertoires nécessaires pour stocker les modèles."""

    for directory in [settings.paths.models_dir / "whisper", settings.paths.models_dir / "embeddings"]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Point d'entrée principal du script."""

    prepare_directories()
    download_spacy(settings.models.spacy_model)
    download_whisper(settings.models.whisper_model_size)
    logger.info("Téléchargement des modèles terminé.")


if __name__ == "__main__":
    main()
