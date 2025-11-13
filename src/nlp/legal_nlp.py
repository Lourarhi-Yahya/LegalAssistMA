"""Module NLP juridique pour LegalAssistMA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import spacy
from src.utils.logger import logger
from transformers import Pipeline, pipeline

from config import settings


@dataclass
class EntityResult:
    """Représente une entité nommée extraite du texte."""

    text: str
    label: str
    start: int
    end: int


@dataclass
class NLPReport:
    """Rassemble les résultats NLP pour un segment ou un document."""

    entities: List[EntityResult]
    sentiment: str
    sentiment_score: float
    category: str
    category_scores: Dict[str, float]
    keywords: List[str]


class LegalNLPProcessor:
    """Enveloppe des fonctionnalités NLP adaptées au contexte juridique marocain."""

    def __init__(self) -> None:
        logger.info("Chargement du modèle spaCy %s", settings.models.spacy_model)
        self.spacy_nlp = spacy.load(settings.models.spacy_model)
        self.sentiment_model_name = "akhooli/bert-base-arabic-camelbert-da-sentiment"
        self.zero_shot_model_name = "joeddav/xlm-roberta-large-xnli"
        self.sentiment_analyzer = self._create_pipeline(
            "sentiment-analysis", self.sentiment_model_name
        )
        self.classifier = self._create_pipeline(
            "zero-shot-classification", self.zero_shot_model_name
        )
        self.legal_labels = ["penal", "civil", "famille", "travail"]

    def analyse(self, text: str) -> NLPReport:
        """Produit un rapport NLP complet pour un texte donné."""

        doc = self.spacy_nlp(text)
        entities = [
            EntityResult(ent.text, ent.label_, int(ent.start_char), int(ent.end_char))
            for ent in doc.ents
        ]
        sentiment_label, sentiment_score = self._analyse_sentiment(text)
        category, category_scores = self._classify_case(text)
        keywords = self._extract_keywords(doc)
        return NLPReport(
            entities=entities,
            sentiment=sentiment_label,
            sentiment_score=sentiment_score,
            category=category,
            category_scores=category_scores,
            keywords=keywords,
        )

    def _analyse_sentiment(self, text: str) -> tuple[str, float]:
        """Retourne le sentiment dominant et son score associé."""

        result = self.sentiment_analyzer(text)[0]
        label = result["label"].lower()
        score = float(result["score"])
        logger.debug("Sentiment détecté: %s (%.2f)", label, score)
        return label, score

    def _classify_case(self, text: str) -> tuple[str, Dict[str, float]]:
        """Classe l'affaire parmi les catégories juridiques définies."""

        result = self.classifier(text, candidate_labels=self.legal_labels)
        scores = {
            label: float(score) for label, score in zip(result["labels"], result["scores"])
        }
        logger.debug("Catégorie prédominante: %s", result["labels"][0])
        return result["labels"][0], scores

    def _extract_keywords(self, doc: spacy.tokens.Doc, max_keywords: int = 12) -> List[str]:
        """Extrait des mots-clés basés sur la morphologie et la fréquence."""

        candidates = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop and len(token) > 3
        ]
        unique_keywords: List[str] = []
        for keyword in candidates:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
            if len(unique_keywords) >= max_keywords:
                break
        logger.debug("%s mots-clés extraits", len(unique_keywords))
        return unique_keywords

    def _create_pipeline(self, task: str, model_name: str) -> Pipeline:
        """Construit un pipeline Transformers avec gestion des erreurs."""

        try:
            return pipeline(
                task,
                model=model_name,
                tokenizer=model_name,
                use_auth_token=settings.api.huggingface_token,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Impossible de charger le pipeline %s (%s)", task, exc)
            raise


def batch_analyse(texts: Sequence[str]) -> List[NLPReport]:
    """Analyse plusieurs textes et retourne les rapports correspondants."""

    processor = LegalNLPProcessor()
    return [processor.analyse(text) for text in texts]
