"""Script de test pour exécuter le pipeline complet sur un fichier audio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from src.pipeline.main_pipeline import MainPipeline


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""

    parser = argparse.ArgumentParser(description="Test du pipeline LegalAssistMA")
    parser.add_argument("audio", type=Path, help="Chemin du fichier audio à traiter")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée du script."""

    args = parse_args()
    pipeline = MainPipeline()
    result = pipeline.process_audio(args.audio)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Pipeline exécuté avec succès")


if __name__ == "__main__":
    main()
