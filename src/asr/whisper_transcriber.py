"""Module de transcription audio basé sur Whisper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import whisper
from src.utils.logger import logger

from config import settings


@dataclass
class TranscriptSegment:
    """Segment de transcription retourné par Whisper."""

    text: str
    start: float
    end: float
    confidence: Optional[float]


class WhisperTranscriber:
    """Encapsule le chargement du modèle Whisper et la transcription."""

    def __init__(self, model_size: Optional[str] = None) -> None:
        self.model_size = model_size or settings.models.whisper_model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Chargement du modèle Whisper %s sur %s", self.model_size, self.device)
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Échec de chargement du modèle Whisper: %s", exc)
            raise

    def transcribe(self, audio_file: Path, language: str = "ar") -> List[TranscriptSegment]:
        """Transcrit un fichier audio et retourne les segments reconnus."""

        try:
            logger.info("Transcription de %s", audio_file)
            result = self.model.transcribe(
                str(audio_file), language=language, fp16=self.device == "cuda"
            )
            segments = [
                TranscriptSegment(
                    text=segment.get("text", "").strip(),
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    confidence=segment.get("avg_logprob"),
                )
                for segment in result.get("segments", [])
            ]
            logger.info("Transcription terminée: %s segments", len(segments))
            return segments
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur durant la transcription: %s", exc)
            raise
