"""Téléchargement des modèles nécessaires au projet."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from subprocess import CalledProcessError

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.utils.logger import logger

from config import settings


def download_whisper(model_size: str) -> None:
    """Télécharge le modèle Whisper sélectionné via l'API openai-whisper."""

    logger.info("Téléchargement du modèle Whisper %s", model_size)
    try:
        subprocess.run(
            ["whisper", "--model", model_size, "--list-models"],
            check=True,
        )
    except FileNotFoundError as error:
        logger.error("Commande whisper introuvable: %s", error)
        raise
    except CalledProcessError as error:
        logger.error("Échec du téléchargement du modèle Whisper: %s", error)
        raise


def ensure_spacy_installed() -> None:
    """Garantit que spaCy est installé avant de lancer le téléchargement."""

    try:
        import importlib.util

        if importlib.util.find_spec("spacy") is None:
            raise ModuleNotFoundError
    except ModuleNotFoundError:
        logger.warning(
            "spaCy est absent. Installation automatique de spacy==3.7.4..."
        )
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "spacy==3.7.4"],
                check=True,
            )
        except CalledProcessError as error:
            logger.error("Impossible d'installer spaCy automatiquement: %s", error)
            raise


def download_spacy(model_name: str) -> None:
    """Télécharge le modèle spaCy requis."""

    ensure_spacy_installed()
    logger.info("Téléchargement du modèle spaCy %s", model_name)
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
        )
    except FileNotFoundError as error:
        logger.error("Python introuvable pour le téléchargement du modèle spaCy: %s", error)
        raise
    except CalledProcessError as error:
        logger.error(
            "Échec du téléchargement du modèle spaCy: %s.",
            error,
        )
        raise


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
