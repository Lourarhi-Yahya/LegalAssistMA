# LegalAssistMA

LegalAssistMA est un système d'assistance juridique basé sur l'intelligence artificielle destiné aux tribunaux marocains. Il couvre la transcription automatique, la diarisation, l'analyse NLP, la recherche d'articles de loi et la génération de recommandations via LLM.

## Fonctionnalités principales

- Prétraitement audio (normalisation 16kHz, réduction de bruit, découpage)
- Transcription automatique Darija (OpenAI Whisper)
- Diarisation des locuteurs (pyannote.audio)
- Analyse NLP juridique (spaCy, Transformers, AraBERT)
- Recherche augmentée par IA (FAISS + Sentence-Transformers)
- Génération de résumés et recommandations (GPT-3.5-turbo)
- API REST (FastAPI) et interface web responsive

## Prérequis système

- Python 3.10 ou 3.11
- 16 Go RAM minimum
- GPU NVIDIA (optionnel, recommandé pour Whisper large-v3)
- ffmpeg installé (pour pydub/Whisper)

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/votre-utilisateur/LegalAssistMA.git
   cd LegalAssistMA
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows : .venv\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement**
   ```bash
   cp .env.example .env
   # Éditer .env pour renseigner OPENAI_API_KEY et HUGGINGFACE_TOKEN
   ```

5. **Télécharger les modèles nécessaires**
   ```bash
   python scripts/download_models.py
   python scripts/create_sample_corpus.py  # génère un mini corpus si besoin
   ```

## Utilisation

### Lancer l'API

```bash
python scripts/run_server.py
```

- Endpoint de santé : `GET http://localhost:8000/health`
- Endpoint principal : `POST http://localhost:8000/transcribe`

### Tester le pipeline en ligne de commande

```bash
python scripts/test_pipeline.py data/audio_samples/exemple.wav
```

### Interface web

Ouvrir `frontend/index.html` dans un navigateur moderne. Configurer le proxy ou servir le frontend pour qu'il pointe vers l'API (par défaut même origine).

## Dépannage

- **Erreur de clé API** : vérifier `OPENAI_API_KEY` dans `.env`.
- **Modèle Pyannote** : nécessite un token Hugging Face valide (`HUGGINGFACE_TOKEN`).
- **ffmpeg manquant** : installer via le gestionnaire de paquets (`sudo apt install ffmpeg`).
- **Mémoire GPU insuffisante** : utiliser un modèle Whisper plus petit (`base` ou `small`).

## Licence

Projet académique – usage éducatif uniquement.
