# Architecture Logicielle

Ce document décrit l'architecture du MVP LegalAssistMA.

## Vue d'ensemble

```
Audio → Prétraitement → Transcription (Whisper) → Diarisation (pyannote) → NLP (spaCy/Transformers)
     → RAG (FAISS + Sentence-Transformers) → LLM (GPT-3.5-turbo) → Rapport JSON/API/Frontend
```

## Composants

- **Prétraitement (`src/preprocessing/audio_processor.py`)** : normalise, nettoie et découpe l'audio.
- **ASR (`src/asr/whisper_transcriber.py`)** : transcrit l'audio en Darija via Whisper.
- **Diarisation (`src/asr/speaker_diarizer.py`)** : identifie les locuteurs et fusionne avec la transcription.
- **NLP (`src/nlp/legal_nlp.py`)** : extraction d'entités, sentiment, mots-clés et classification.
- **RAG (`src/rag/legal_rag.py`)** : recherche des articles de loi via embeddings et FAISS.
- **LLM (`src/nlp/llm_generator.py`)** : produit résumé et recommandations avec GPT-3.5-turbo.
- **Pipeline (`src/pipeline/main_pipeline.py`)** : orchestre l'ensemble du flux et sauvegarde les résultats.
- **API (`src/api/main.py`)** : expose les endpoints REST.
- **Frontend (`frontend/index.html`)** : interface pour charger un audio et visualiser le rapport.

## Données et stockage

- Audio d'entrée : `data/audio_samples/`
- Audio prétraité : `data/processed_audio/`
- Corpus juridique : `data/corpus/legal_corpus.json`
- Résultats : `data/outputs/`
- Modèles : `models/`
- Logs : `logs/app.log`

## Flux détaillé

1. **Upload Audio** : via API ou script.
2. **Prétraitement** : conversion en 16kHz mono, réduction de bruit, découpage en 30s.
3. **Transcription** : Whisper transcrit chaque chunk (ajustement des timestamps).
4. **Diarisation** : pyannote associe un locuteur à chaque segment.
5. **NLP** : spaCy & Transformers extraient entités, sentiment, catégorie, mots-clés.
6. **RAG** : FAISS identifie les 5 articles les plus pertinents.
7. **LLM** : GPT-3.5 synthétise un résumé et des recommandations.
8. **Persist** : Sauvegarde JSON et renvoi via API.

## Sécurité et limites

- Clés API gérées via `.env`.
- Limites sur la durée audio configurables dans `config.py`.
- Logs via Loguru pour audit.

## Prochaines étapes

- Tests unitaires et intégration continue.
- Gestion avancée des erreurs métier.
- Optimisations GPU et cache des modèles.
