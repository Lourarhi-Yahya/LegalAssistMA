"""Configuration centrale pour LegalAssistMA."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - dépendance optionnelle
    def load_dotenv(*_: Any, **__: Any) -> bool:
        """Substitut minimal lorsque python-dotenv est absent."""

        return False

load_dotenv()


@dataclass(frozen=True)
class PathConfig:
    """Paramètres des répertoires utilisés dans tout le projet."""

    base_dir: Path = Path(os.getenv("PROJECT_ROOT", Path(__file__).parent))
    data_dir: Path = Path(os.getenv("DATA_DIR", base_dir / "data"))
    models_dir: Path = Path(os.getenv("MODELS_DIR", base_dir / "models"))
    log_file: Path = Path(os.getenv("LOG_FILE", base_dir / "logs" / "app.log"))


@dataclass(frozen=True)
class ModelConfig:
    """Paramètres liés aux modèles de transcription, NLP et embeddings."""

    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
    pyannote_pipeline: str = os.getenv(
        "PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1"
    )
    sentence_embedding_model: str = os.getenv(
        "SENTENCE_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    spacy_model: str = os.getenv("SPACY_MODEL", "fr_core_news_md")


@dataclass(frozen=True)
class LimitsConfig:
    """Limites opérationnelles pour protéger l'API et le pipeline."""

    max_audio_minutes: int = int(os.getenv("MAX_AUDIO_DURATION_MINUTES", "60"))
    max_transcript_characters: int = int(
        os.getenv("MAX_TRANSCRIPT_CHARACTERS", "20000")
    )


@dataclass(frozen=True)
class APIConfig:
    """Clés API et secrets nécessaires."""

    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")


@dataclass(frozen=True)
class Settings:
    """Objet de configuration global regroupant tous les paramètres."""

    paths: PathConfig = PathConfig()
    models: ModelConfig = ModelConfig()
    limits: LimitsConfig = LimitsConfig()
    api: APIConfig = APIConfig()


settings = Settings()

