"""Système RAG pour la recherche d'articles de loi pertinents."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from src.utils.logger import logger
from sentence_transformers import SentenceTransformer

from config import settings


@dataclass
class LegalArticle:
    """Représente un article de loi du corpus."""

    code: str
    article: str
    text: str
    category: str
    keywords: List[str]


class LegalRAG:
    """Construit un index FAISS et effectue des recherches sémantiques."""

    def __init__(self, corpus_path: Path | None = None) -> None:
        self.corpus_path = corpus_path or (settings.paths.data_dir / "corpus" / "legal_corpus.json")
        self.articles: List[LegalArticle] = []
        self.embeddings: np.ndarray | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.model = self._load_model()
        self._load_corpus()
        self._build_index()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[LegalArticle, float]]:
        """Recherche les articles les plus pertinents pour la requête donnée."""

        if not query.strip():
            raise ValueError("La requête de recherche ne peut pas être vide.")
        if self.index is None or self.embeddings is None:
            raise RuntimeError("L'index FAISS n'a pas été initialisé.")

        query_embedding = self._embed_texts([query])
        scores, indices = self.index.search(query_embedding, top_k)
        results: List[Tuple[LegalArticle, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            article = self.articles[int(idx)]
            results.append((article, float(score)))
        logger.debug("Recherche RAG renvoie %s résultats", len(results))
        return results

    def _load_model(self) -> SentenceTransformer:
        """Charge le modèle d'embedding sentence-transformers."""

        try:
            return SentenceTransformer(
                settings.models.sentence_embedding_model,
                cache_folder=str(settings.paths.models_dir / "embeddings"),
                use_auth_token=settings.api.huggingface_token,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Impossible de charger le modèle d'embedding: %s", exc)
            raise

    def _load_corpus(self) -> None:
        """Lit le corpus JSON depuis le disque et crée les objets articles."""

        try:
            with open(self.corpus_path, "r", encoding="utf-8") as corpus_file:
                entries = json.load(corpus_file)
        except FileNotFoundError as exc:
            logger.error("Corpus introuvable: %s", exc)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Format de corpus invalide: %s", exc)
            raise

        self.articles = [
            LegalArticle(
                code=str(entry.get("code", "")),
                article=str(entry.get("article", "")),
                text=str(entry.get("text", "")),
                category=str(entry.get("category", "")),
                keywords=list(entry.get("keywords", [])),
            )
            for entry in entries
        ]
        logger.info("%s articles juridiques chargés", len(self.articles))

    def _build_index(self) -> None:
        """Construit l'index FAISS en mémoire à partir des embeddings."""

        if not self.articles:
            raise ValueError("Aucun article chargé pour construire l'index.")
        texts = [article.text for article in self.articles]
        self.embeddings = self._embed_texts(texts)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        logger.info("Index FAISS construit avec dimension %s", dimension)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Génère les embeddings normalisés pour une liste de textes."""

        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vectors, dtype=np.float32)
