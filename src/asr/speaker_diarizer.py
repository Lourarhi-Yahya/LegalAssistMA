"""Module de diarisation des locuteurs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from src.utils.logger import logger
from pyannote.audio import Pipeline

from config import settings
from src.asr.whisper_transcriber import TranscriptSegment


@dataclass
class SpeakerSegment:
    """Association d'un texte et d'un locuteur identifié."""

    speaker: str
    text: str
    start: float
    end: float


class SpeakerDiarizer:
    """Encapsule l'utilisation de pyannote.audio pour la diarisation."""

    def __init__(self, pipeline_name: Optional[str] = None) -> None:
        self.pipeline_name = pipeline_name or settings.models.pyannote_pipeline
        logger.info("Chargement du pipeline de diarisation %s", self.pipeline_name)
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.pipeline_name, use_auth_token=settings.api.huggingface_token
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Échec de chargement du pipeline Pyannote: %s", exc)
            raise

    def diarize(
        self, audio_file: Path, transcripts: Iterable[TranscriptSegment]
    ) -> List[SpeakerSegment]:
        """Effectue la diarisation puis fusionne avec la transcription."""

        try:
            diarization = self.pipeline(str(audio_file))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur lors de la diarisation: %s", exc)
            raise

        speaker_segments = list(diarization.itertracks(yield_label=True))
        logger.debug("%s segments de locuteurs détectés", len(speaker_segments))
        return self._merge_transcripts(transcripts, speaker_segments)

    def _merge_transcripts(
        self,
        transcripts: Iterable[TranscriptSegment],
        speaker_segments: List[tuple],
    ) -> List[SpeakerSegment]:
        """Associe chaque segment de transcription au locuteur dominant."""

        merged: List[SpeakerSegment] = []
        for segment in transcripts:
            speaker = self._find_speaker_for_segment(segment, speaker_segments)
            merged.append(
                SpeakerSegment(
                    speaker=speaker,
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                )
            )
        logger.info("Fusion transcription/diarisation générée (%s segments)", len(merged))
        return merged

    def _find_speaker_for_segment(
        self, segment: TranscriptSegment, speaker_segments: List[tuple]
    ) -> str:
        """Détermine le locuteur qui recouvre le plus le segment donné."""

        best_overlap = 0.0
        best_speaker = "inconnu"
        for (time_segment, track, label) in speaker_segments:
            overlap = self._compute_overlap(segment, time_segment)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = label or track
        return str(best_speaker)

    @staticmethod
    def _compute_overlap(segment: TranscriptSegment, time_segment) -> float:
        """Calcule la durée d'intersection entre deux segments."""

        start = max(segment.start, time_segment.start)
        end = min(segment.end, time_segment.end)
        return max(0.0, end - start)
