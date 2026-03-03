"""
03_prepare_dataset.py — version corrigée
=========================================
Corrections apportées :
  1. Chemins Windows corrects (raw strings)
  2. whisper-small au lieu de whisper-base (cohérent avec l'entraînement)
  3. CSV_PATH pointe vers dataset_concat.csv (sortie de 09_concat_audio.py)
  4. batch_size=8 pour éviter les crashes mémoire
"""

import os
import librosa
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor

# ==============================
# CONFIG
# ==============================

BASE = r"C:\Users\MSI\Downloads\DATA prete-20260203T090516Z-3-001"

CSV_PATH = os.path.join(BASE, "data", "dataset_concat.csv")   # ← sortie de 09_concat_audio.py
OUTPUT_DIR = os.path.join(BASE, "data", "whisper_dataset_v2")   # ← nouveau dossier

MODEL_NAME = "openai/whisper-small"   # ← small, pas base (cohérent avec train)

# ==============================
# VÉRIFICATIONS
# ==============================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Le fichier {CSV_PATH} n'existe pas.\n"
        f"Lance d'abord : python 00_clean_audio_fixed.py\n"
        f"Puis           : python 09_concat_audio.py"
    )

# ==============================
# PROCESSOR
# ==============================

print(f"Chargement processeur : {MODEL_NAME}")
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="fr",
    task="transcribe",
)

# ==============================
# CHARGER CSV
# ==============================

print(f"Chargement CSV : {CSV_PATH}")
dataset = load_dataset("csv", data_files=CSV_PATH)["train"]
print(f"Total exemples : {len(dataset)}")

# ==============================
# FONCTION PRÉPARATION
# ==============================


def prepare(batch):
    audio_arrays = []
    for path in batch["audio_path"]:
        try:
            samples, _ = librosa.load(path, sr=16000, mono=True)
            audio_arrays.append(samples)
        except Exception as e:
            raise RuntimeError(f"Erreur audio {path}: {e}")

    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="fr", task="transcribe")
    lang_task_tokens = [token_id for token_id, _ in forced_decoder_ids]

    text_labels = processor.tokenizer(
        batch["transcription"],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    eos_token_id = processor.tokenizer.eos_token_id
    full_labels = []
    for text_ids in text_labels.input_ids:
        text_tokens = [
            t for t in text_ids.tolist()
            if t != processor.tokenizer.pad_token_id and t >= 0
        ]
        full_labels.append(lang_task_tokens + text_tokens + [eos_token_id])

    batch["input_features"] = [feat.numpy() for feat in inputs.input_features]
    batch["labels"] = full_labels
    return batch

# ==============================
# TRAITEMENT
# ==============================


print("Traitement des audios...")
dataset = dataset.map(
    prepare,
    remove_columns=[c for c in dataset.column_names if c not in ("input_features", "labels")],
    batched=True,
    batch_size=8,
)

# ==============================
# SPLIT
# ==============================

dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)

final_dataset = DatasetDict({
    "train": train_val["train"],
    "validation": train_val["test"],
    "test": dataset["test"],
})

print(f"\nDataset final :\n{final_dataset}")

# ==============================
# SAUVEGARDER
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
final_dataset.save_to_disk(OUTPUT_DIR)

print(f"\n✅ Dataset sauvegardé : {OUTPUT_DIR}")
print(f"\nProchaine étape — dans 04_train_whisper_v8.py, mettre :")
print(f'  DATASET_PATH = r"{OUTPUT_DIR}"')
