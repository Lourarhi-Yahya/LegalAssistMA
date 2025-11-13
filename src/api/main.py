"""Entrée FastAPI pour le système LegalAssistMA."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.pipeline.main_pipeline import MainPipeline

app = FastAPI(title="LegalAssistMA", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = MainPipeline()


@app.get("/health")
def health_check() -> dict[str, str]:
    """Vérifie que l'API fonctionne correctement."""

    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict:
    """Transcrit un fichier audio téléchargé et retourne le rapport complet."""

    if file.content_type not in {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/ogg"}:
        raise HTTPException(status_code=400, detail="Format audio non supporté.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = Path(tmp.name)
        logger.info("Fichier reçu %s (%s octets)", file.filename, len(content))
        result = pipeline.process_audio(temp_path)
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erreur durant le traitement de %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")
    finally:
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink(missing_ok=True)
