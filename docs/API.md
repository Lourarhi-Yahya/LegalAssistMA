# Documentation API

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET `/health`

- **Description** : Vérifie l'état de l'API.
- **Réponse (200)** :
  ```json
  {"status": "ok"}
  ```

### POST `/transcribe`

- **Description** : Téléverse un fichier audio et retourne le rapport complet.
- **Paramètres** :
  - `file` : fichier audio (WAV, MP3, OGG, MPEG)
- **Réponse (200)** :
  ```json
  {
    "transcription": [
      {"text": "...", "start": 0.0, "end": 3.2, "confidence": -0.12}
    ],
    "diarization": [
      {"speaker": "SPEAKER_00", "text": "...", "start": 0.0, "end": 3.2}
    ],
    "nlp_report": {
      "entities": [{"text": "Rabat", "label": "LOC", "start": 12, "end": 17}],
      "sentiment": "neutral",
      "sentiment_score": 0.62,
      "category": "penal",
      "category_scores": {"penal": 0.62, "civil": 0.21, "famille": 0.1, "travail": 0.07},
      "keywords": ["violence", "plainte", "agression"]
    },
    "legal_articles": [
      {
        "code": "Code Pénal Marocain",
        "article": "400",
        "text": "Quiconque, volontairement...",
        "category": "penal",
        "keywords": ["coups", "blessures", "violence"],
        "score": 0.81
      }
    ],
    "llm_result": {
      "summary": "- Fait : ...",
      "recommendations": [
        {"texte": "Informer la victime des délais de prescription (confiance : haute)"}
      ]
    }
  }
  ```
- **Erreurs possibles** :
  - `400` : format audio non supporté.
  - `413` : fichier trop volumineux (si implémenté).
  - `500` : erreur interne du serveur.

## Utilisation

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Swagger & ReDoc

- Swagger UI : `http://localhost:8000/docs`
- ReDoc : `http://localhost:8000/redoc`
