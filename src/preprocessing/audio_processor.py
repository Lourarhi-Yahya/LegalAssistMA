"""Module de prétraitement audio pour LegalAssistMA."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import soundfile as sf
from loguru import logger
from noisereduce import reduce_noise
from pydub import AudioSegment

from config import settings


@dataclass
class AudioChunk:
    """Représente un segment audio découpé après prétraitement."""

    file_path: Path
    start_time: float
    end_time: float


class AudioProcessor:
    """Effectue le chargement, la normalisation et le découpage des fichiers audio."""

    def __init__(self, target_sr: int = 16_000, chunk_duration: int = 30) -> None:
        self.target_sr = target_sr
        self.chunk_duration = chunk_duration
        self.output_dir = settings.paths.data_dir / "processed_audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, audio_path: Path) -> List[AudioChunk]:
        """Traite un fichier audio et retourne les chunks nettoyés.

        Args:
            audio_path: Chemin du fichier audio original.

        Returns:
            Liste de segments prêts pour la transcription.
        """

        try:
            logger.info("Prétraitement de l'audio %s", audio_path)
            audio_segment = self._load_audio(audio_path)
            cleaned_audio = self._denoise_audio(audio_segment)
            chunks = self._split_audio(cleaned_audio)
            logger.info("Prétraitement terminé: %s segments produits", len(chunks))
            return chunks
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur lors du prétraitement: %s", exc)
            raise

    def _load_audio(self, audio_path: Path) -> AudioSegment:
        """Charge le fichier audio en assurant un format cohérent."""

        try:
            audio = AudioSegment.from_file(audio_path)
            normalized = audio.set_frame_rate(self.target_sr).set_channels(1)
            logger.debug("Audio chargé et normalisé en %s Hz mono", self.target_sr)
            return normalized
        except Exception as exc:  # noqa: BLE001
            logger.error("Impossible de charger %s : %s", audio_path, exc)
            raise

    def _denoise_audio(self, audio_segment: AudioSegment) -> AudioSegment:
        """Réduit le bruit de fond tout en préservant l'information vocale."""

        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        if samples.size == 0:
            raise ValueError("Le fichier audio ne contient aucun échantillon.")
        logger.debug("Réduction de bruit sur %s échantillons", samples.size)
        reduced = reduce_noise(y=samples, sr=self.target_sr)
        output_segment = audio_segment._spawn(np.int16(reduced).tobytes())
        return output_segment

    def _split_audio(self, audio_segment: AudioSegment) -> List[AudioChunk]:
        """Découpe l'audio en segments fixes et les sauvegarde sur disque."""

        chunk_length_ms = self.chunk_duration * 1000
        total_duration_ms = len(audio_segment)
        chunks: List[AudioChunk] = []
        start_ms = 0
        index = 0
        while start_ms < total_duration_ms:
            end_ms = min(start_ms + chunk_length_ms, total_duration_ms)
            chunk = audio_segment[start_ms:end_ms]
            chunk_path = self._export_chunk(chunk, index)
            chunks.append(
                AudioChunk(
                    file_path=chunk_path,
                    start_time=start_ms / 1000,
                    end_time=end_ms / 1000,
                )
            )
            start_ms = end_ms
            index += 1
        return chunks

    def _export_chunk(self, chunk: AudioSegment, index: int) -> Path:
        """Exporte un segment audio au format WAV 16kHz mono."""

        buffer = io.BytesIO()
        chunk.export(buffer, format="wav")
        buffer.seek(0)
        file_path = self.output_dir / f"chunk_{index:04d}.wav"
        data, samplerate = sf.read(buffer)
        sf.write(file_path, data, samplerate)
        logger.debug("Chunk %s sauvegardé dans %s", index, file_path)
        return file_path


def process_audio_files(audio_files: Sequence[Path]) -> List[List[AudioChunk]]:
    """Fonction utilitaire pour traiter une liste de fichiers audio."""

    processor = AudioProcessor()
    results: List[List[AudioChunk]] = []
    for audio_file in audio_files:
        results.append(processor.process(audio_file))
    return results
