"""Générateur de résumés et recommandations via l'API OpenAI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.utils.logger import logger
from openai import OpenAI

from config import settings
from src.asr.speaker_diarizer import SpeakerSegment
from src.rag.legal_rag import LegalArticle


@dataclass
class LLMResult:
    """Structure de sortie pour les réponses du LLM."""

    summary: str
    recommendations: List[Dict[str, str]]


class LLMGenerator:
    """Interagit avec GPT-3.5-turbo pour créer un rapport juridique exploitable."""

    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        if not settings.api.openai_api_key:
            raise ValueError("La clé API OpenAI est requise pour utiliser le LLM.")
        self.client = OpenAI(api_key=settings.api.openai_api_key)
        self.model_name = model_name

    def build_report(
        self,
        transcript: List[SpeakerSegment],
        legal_articles: List[LegalArticle],
        nlp_summary: str,
    ) -> LLMResult:
        """Génère un résumé global et des recommandations détaillées."""

        prompt = self._compose_prompt(transcript, legal_articles, nlp_summary)
        logger.info("Requête au modèle %s pour la génération de rapport", self.model_name)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant juridique spécialisé dans le droit marocain. "
                        "Fourni des réponses structurées, factuelles et prudentes."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        message = completion.choices[0].message.content
        summary, recommendations = self._post_process(message)
        return LLMResult(summary=summary, recommendations=recommendations)

    def _compose_prompt(
        self,
        transcript: List[SpeakerSegment],
        legal_articles: List[LegalArticle],
        nlp_summary: str,
    ) -> str:
        """Assemble les différentes informations pour un prompt cohérent."""

        transcript_text = "\n".join(
            f"[{segment.start:.2f}-{segment.end:.2f}] {segment.speaker}: {segment.text}"
            for segment in transcript
        )
        articles_text = "\n".join(
            f"Article {article.article} ({article.code}) - {article.text[:400]}..."
            for article in legal_articles
        )
        prompt = (
            "Contexte NLP:\n" + nlp_summary + "\n\n"
            "Transcription diarisée:\n" + transcript_text + "\n\n"
            "Articles pertinents:\n" + articles_text + "\n\n"
            "1. Fournis un résumé structuré en 5 puces max (faits, parties, décisions, preuves, ton).\n"
            "2. Fournis 3 recommandations juridiques pratiques avec estimation de confiance (faible/moyenne/haute).\n"
            "3. Signale les incertitudes ou données manquantes."
        )
        return prompt

    def _post_process(self, message: str) -> tuple[str, List[Dict[str, str]]]:
        """Sépare le résumé des recommandations en supposant un format markdown."""

        sections = message.split("Recommandations")
        summary = sections[0].strip()
        recommendations: List[Dict[str, str]] = []
        if len(sections) > 1:
            for line in sections[1].splitlines():
                if "-" not in line:
                    continue
                parts = line.split("-", maxsplit=1)
                recommendation = parts[1].strip()
                recommendations.append({"texte": recommendation})
        if not recommendations:
            recommendations.append({"texte": "Aucune recommandation explicite fournie."})
        return summary, recommendations
