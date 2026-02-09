import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du device: {device}")

DATASET_PATH = "data/whisper_dataset"
MODEL_OUTPUT_DIR = "model/whisper-medical-fr"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Le dataset {DATASET_PATH} n'existe pas. Exécutez d'abord 03_prepare_dataset.py")

try:
    dataset = load_from_disk(DATASET_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du dataset: {e}")

try:
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        language="fr",
        task="transcribe"
    )
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du processeur: {e}")

try:
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    ).to(device)
    # CRITIQUE: Forcer la langue française et la tâche de transcription
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="fr",
        task="transcribe"
    )
    model.config.forced_decoder_ids = forced_decoder_ids
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")


@dataclass
class SimpleWhisperDataCollator:
    """
    Data collator basique pour notre dataset déjà pré-tokenisé.
    Il empile les features audio et pad les labels à -100.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_tensors: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []

        expected_len = 3000  # longueur temporelle attendue par Whisper

        for f in features:
            feat = torch.tensor(f["input_features"])  # [80, T] ou [80, T']

            # S'assurer que la dimension temporelle est à 3000 (pad ou tronque)
            if feat.shape[-1] < expected_len:
                pad_width = expected_len - feat.shape[-1]
                pad = torch.zeros(feat.shape[0], pad_width, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=-1)
            elif feat.shape[-1] > expected_len:
                feat = feat[:, :expected_len]

            input_tensors.append(feat)

            lab = torch.tensor(f["labels"])
            label_tensors.append(lab)

        batch_input_features = torch.stack(input_tensors)  # [B, 80, 3000]
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            label_tensors, batch_first=True, padding_value=-100
        )

        return {
            "input_features": batch_input_features,
            "labels": batch_labels,
        }


data_collator = SimpleWhisperDataCollator()

training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    # Learning rate optimal pour fine-tuning Whisper (pas trop petit, pas trop grand)
    learning_rate=1e-5,
    # Warmup pour stabiliser l'entraînement au début
    warmup_steps=500,
    # Nombre d'époques raisonnable pour un dataset de ~720 exemples
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),
    save_total_limit=3,
    predict_with_generate=True,
    logging_steps=50,  # Afficher les logs régulièrement
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)

try:
    trainer.train()
except Exception as e:
    raise RuntimeError(f"Erreur lors de l'entraînement: {e}")

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
model.save_pretrained(MODEL_OUTPUT_DIR)
processor.save_pretrained(MODEL_OUTPUT_DIR)

print("✅ Modèle fine-tuné et sauvegardé")
