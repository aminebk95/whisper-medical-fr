# Guide de Test

Ce guide explique comment tester chaque script dans l'ordre.

## Prérequis

Assurez-vous d'avoir installé les dépendances :
```bash
pip install torch torchaudio transformers datasets librosa striprtf
```

## Ordre d'exécution

### 1. Conversion MP3 → WAV
```bash
python 01_mp3_to_wav.py
```
**Vérification :**
- Le dossier `data/wav/` doit être créé
- Des fichiers `.wav` doivent apparaître dans `data/wav/`
- Le script affiche le nombre de fichiers convertis

### 2. Création du CSV
```bash
python 02_rtf_to_csv.py
```
**Vérification :**
- Le fichier `data/dataset.csv` doit être créé
- Ouvrir le CSV pour vérifier qu'il contient les colonnes : `id`, `audio_path`, `transcription`
- Vérifier que le nombre de phrases correspond aux lignes du CSV

### 3. Préparation du dataset
```bash
python 03_prepare_dataset.py
```
**Vérification :**
- Le dossier `data/whisper_dataset/` doit être créé
- Le script doit afficher "✅ Dataset prêt et sauvegardé"
- Vérifier que le dossier contient les sous-dossiers `train`, `validation`, `test`

### 4. Entraînement du modèle
```bash
python 04_train_whisper.py
```
**Note :** Cette étape peut prendre plusieurs heures selon votre matériel.
**Vérification :**
- Le dossier `model/whisper-medical-fr/` doit être créé
- Des checkpoints doivent apparaître pendant l'entraînement
- Le script affiche les métriques d'entraînement

### 5. Test du modèle
```bash
python 05_test_model.py
```
**Vérification :**
- Le script doit afficher une transcription
- Vérifier que la transcription est cohérente avec l'audio testé

## Test rapide (sans entraînement complet)

Pour tester rapidement sans faire l'entraînement complet :

1. Exécutez les scripts 1, 2 et 3
2. Pour tester le modèle, utilisez le modèle pré-entraîné :
   ```python
   # Modifier 05_test_model.py temporairement
   MODEL_PATH = "openai/whisper-small"  # au lieu de "model/whisper-medical-fr"
   ```

## Dépannage

### Erreur "FileNotFoundError"
- Vérifiez que les fichiers/dossiers existent
- Vérifiez les chemins dans les scripts

### Erreur "CUDA out of memory"
- Réduisez `per_device_train_batch_size` dans `04_train_whisper.py`
- Augmentez `gradient_accumulation_steps`

### Erreur lors de la conversion audio
- Vérifiez que les fichiers MP3 ne sont pas corrompus
- Vérifiez l'espace disque disponible
