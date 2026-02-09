import os
import librosa
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor

CSV_PATH = "data/dataset.csv"
OUTPUT_DIR = "data/whisper_dataset"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Le fichier {CSV_PATH} n'existe pas. Exécutez d'abord 02_rtf_to_csv.py")

try:
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        language="fr",
        task="transcribe"
    )
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du processeur Whisper: {e}")

try:
    dataset = load_dataset("csv", data_files=CSV_PATH)["train"]
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du dataset CSV: {e}")


def prepare(batch):
    """Charge les fichiers audio à partir des chemins et prépare les features/labels."""
    audio_arrays = []
    for path in batch["audio_path"]:
        try:
            samples, _ = librosa.load(path, sr=16000, mono=True)
            audio_arrays.append(samples)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement audio {path}: {e}")

    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    # Obtenir les tokens de langue et tâche pour le français
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="fr",
        task="transcribe"
    )
    # Les forced_decoder_ids sont une liste de tuples (token_id, token_value)
    # On extrait juste les token_id
    lang_task_tokens = [token_id for token_id, _ in forced_decoder_ids]
    
    # Tokeniser les transcriptions (sans ajouter BOS/EOS automatiquement)
    text_labels = processor.tokenizer(
        batch["transcription"],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,  # On gère nous-mêmes les tokens spéciaux
    )
    
    # Construire les labels complets: [lang_tokens] + [text_tokens] + [EOS]
    # Format attendu par Whisper: [lang, task, text..., eos]
    full_labels = []
    eos_token_id = processor.tokenizer.eos_token_id
    for text_ids in text_labels.input_ids:
        # Enlever les padding tokens (-100 ou 0) pour ne garder que les vrais tokens
        text_tokens = [t for t in text_ids.tolist() if t != processor.tokenizer.pad_token_id and t >= 0]
        # Combiner: [lang_task_tokens] + [text_tokens] + [EOS]
        label = lang_task_tokens + text_tokens + [eos_token_id]
        full_labels.append(label)

    batch["input_features"] = [feat.numpy() for feat in inputs.input_features]
    batch["labels"] = full_labels
    return batch

try:
    dataset = dataset.map(
        prepare,
        remove_columns=dataset.column_names,
        batched=True,
    )
except Exception as e:
    raise RuntimeError(f"Erreur lors de la préparation du dataset: {e}")

dataset = dataset.train_test_split(test_size=0.2)
train_val = dataset["train"].train_test_split(test_size=0.1)

final_dataset = DatasetDict({
    "train": train_val["train"],
    "validation": train_val["test"],
    "test": dataset["test"]
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
final_dataset.save_to_disk(OUTPUT_DIR)

print("✅ Dataset prêt et sauvegardé")
