"""Pipeline principal orchestrant les différentes étapes du système."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from src.utils.logger import logger

from config import settings
from src.asr.speaker_diarizer import SpeakerDiarizer, SpeakerSegment
from src.asr.whisper_transcriber import TranscriptSegment, WhisperTranscriber
from src.nlp.legal_nlp import LegalNLPProcessor
from src.nlp.llm_generator import LLMGenerator, LLMResult
from src.preprocessing.audio_processor import AudioChunk, AudioProcessor
from src.rag.legal_rag import LegalArticle, LegalRAG


@dataclass
class PipelineOutput:
    """Structure finale retournée par le pipeline complet."""

    transcription: List[TranscriptSegment]
    diarization: List[SpeakerSegment]
    nlp_report: Dict
    legal_articles: List[Dict]
    llm_result: LLMResult

    def to_dict(self) -> Dict:
        """Convertit l'objet en dictionnaire sérialisable."""

        return {
            "transcription": [asdict(segment) for segment in self.transcription],
            "diarization": [asdict(segment) for segment in self.diarization],
            "nlp_report": self.nlp_report,
            "legal_articles": self.legal_articles,
            "llm_result": {
                "summary": self.llm_result.summary,
                "recommendations": self.llm_result.recommendations,
            },
        }


class MainPipeline:
    """Gère l'exécution séquentielle de toutes les composantes du système."""

    def __init__(self) -> None:
        self.audio_processor = AudioProcessor()
        self.transcriber = WhisperTranscriber()
        self.diarizer = SpeakerDiarizer()
        self.nlp_processor = LegalNLPProcessor()
        self.rag = LegalRAG()
        self.llm = LLMGenerator()

    def process_audio(self, audio_path: Path) -> PipelineOutput:
        """Exécute le pipeline complet sur un fichier audio unique."""

        logger.info("Démarrage du pipeline pour %s", audio_path)
        self._validate_audio_length(audio_path)
        chunks = self.audio_processor.process(audio_path)
        transcripts = self._transcribe_chunks(chunks)
        diarized = self.diarizer.diarize(audio_path, transcripts)
        nlp_report = self._build_nlp_report(diarized)
        rag_results = self._search_legal_articles(nlp_report)
        llm_result = self._generate_llm_report(diarized, rag_results, nlp_report)
        output = PipelineOutput(
            transcription=transcripts,
            diarization=diarized,
            nlp_report=nlp_report,
            legal_articles=[asdict(article) | {"score": score} for article, score in rag_results],
            llm_result=llm_result,
        )
        self._persist_output(audio_path, output)
        logger.info("Pipeline terminé pour %s", audio_path)
        return output

    def _transcribe_chunks(self, chunks: List[AudioChunk]) -> List[TranscriptSegment]:
        """Transcrit chaque chunk et ajuste les timestamps."""

        transcripts: List[TranscriptSegment] = []
        for chunk in chunks:
            chunk_segments = self.transcriber.transcribe(chunk.file_path)
            for segment in chunk_segments:
                adjusted = TranscriptSegment(
                    text=segment.text,
                    start=segment.start + chunk.start_time,
                    end=segment.end + chunk.start_time,
                    confidence=segment.confidence,
                )
                transcripts.append(adjusted)
        return transcripts

    def _build_nlp_report(self, diarized: List[SpeakerSegment]) -> Dict:
        """Agrège les textes diarises pour créer un rapport NLP global."""

        full_text = "\n".join(f"{segment.speaker}: {segment.text}" for segment in diarized)
        report = self.nlp_processor.analyse(full_text)
        return {
            "entities": [asdict(entity) for entity in report.entities],
            "sentiment": report.sentiment,
            "sentiment_score": report.sentiment_score,
            "category": report.category,
            "category_scores": report.category_scores,
            "keywords": report.keywords,
        }

    def _search_legal_articles(self, nlp_report: Dict) -> List[tuple[LegalArticle, float]]:
        """Utilise les mots-clés et la catégorie pour interroger le RAG."""

        keywords = " ".join(nlp_report.get("keywords", []))
        query = f"{nlp_report.get('category', '')} {keywords}".strip()
        if not query:
            query = "procédure judiciaire"
        return self.rag.search(query)

    def _generate_llm_report(
        self,
        diarized: List[SpeakerSegment],
        rag_results: List[tuple[LegalArticle, float]],
        nlp_report: Dict,
    ) -> LLMResult:
        """Construit les entrées pour le LLM et récupère le rapport final."""

        articles = [article for article, _ in rag_results]
        nlp_summary = json.dumps(nlp_report, ensure_ascii=False, indent=2)
        return self.llm.build_report(diarized, articles, nlp_summary)

    def _persist_output(self, audio_path: Path, output: PipelineOutput) -> None:
        """Sauvegarde les résultats du pipeline dans le dossier outputs."""

        outputs_dir = settings.paths.data_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_file = outputs_dir / f"{audio_path.stem}_report.json"
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(output.to_dict(), file, ensure_ascii=False, indent=2)
        logger.info("Résultats sauvegardés dans %s", output_file)

    def _validate_audio_length(self, audio_path: Path) -> None:
        """Vérifie que la durée du fichier respecte les limites configurées."""

        import soundfile as sf  # import local pour accélérer le chargement global

        with sf.SoundFile(audio_path) as audio_file:
            duration_minutes = audio_file.frames / audio_file.samplerate / 60
        if duration_minutes > settings.limits.max_audio_minutes:
            raise ValueError("Durée audio supérieure à la limite autorisée.")
