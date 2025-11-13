# LegalAssistMA

LegalAssistMA est un système d'assistance juridique basé sur l'IA destiné aux tribunaux marocains. Ce dépôt contiendra le pipeline complet (prétraitement audio, transcription, diarisation, NLP, RAG, LLM) ainsi qu'une API FastAPI et une interface web légère.

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/votre-utilisateur/LegalAssistMA.git
   cd LegalAssistMA
   ```

2. **Créer un environnement virtuel Python 3.10+**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sous Windows : .venv\\Scripts\\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement**
   - Copier le fichier `.env.example` vers `.env`
   - Renseigner les clés API requises (OpenAI, Hugging Face) et ajuster les chemins si nécessaire

5. **Télécharger les modèles** (sera détaillé dans la documentation finale)
   ```bash
   python scripts/download_models.py
   ```

Une fois ces étapes réalisées, vous pourrez exécuter le pipeline et l'API conformément aux instructions qui seront ajoutées dans la suite du projet.
