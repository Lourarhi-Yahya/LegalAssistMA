"""Génère un corpus juridique JSON d'exemple."""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from config import settings

SAMPLE_ARTICLES = [
    {
        "code": "Code Pénal Marocain",
        "article": "400",
        "text": "Quiconque, volontairement, fait des blessures ou porte des coups...",
        "category": "penal",
        "keywords": ["coups", "blessures", "violence"],
    },
    {
        "code": "Code de Procédure Civile Marocain",
        "article": "32",
        "text": "La demande introductive d'instance contient l'exposé sommaire des moyens...",
        "category": "civil",
        "keywords": ["procédure", "instance", "demande"],
    },
]


def main() -> None:
    """Crée le fichier JSON si absent."""

    corpus_dir = settings.paths.data_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = corpus_dir / "legal_corpus.json"
    if corpus_file.exists():
        logger.info("Le corpus existe déjà: %s", corpus_file)
        return
    with open(corpus_file, "w", encoding="utf-8") as file:
        json.dump(SAMPLE_ARTICLES, file, ensure_ascii=False, indent=2)
    logger.info("Corpus d'exemple généré dans %s", corpus_file)


if __name__ == "__main__":
    main()
