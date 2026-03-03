"""
Whisper Training — Windows Safe Version
GTX 1650 4GB + petit dataset
Compatible Transformers récents
"""

import os
import torch
import numpy as np
import evaluate
from dataclasses import dataclass
from typing import Any
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

# ==============================
# CONFIG
# ==============================

BASE = r"C:\Users\MSI\Downloads\DATA prete-20260203T090516Z-3-001"
DATASET_PATH = os.path.join(BASE, "data", "whisper_dataset_v2")
OUTPUT_DIR = os.path.join(BASE, "data", "whisper-medical-fr-v6")

MODEL_NAME = "openai/whisper-small"

FREEZE_N = 4
NUM_EPOCHS = 20
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 1e-5
PATIENCE = 3
MAX_LENGTH = 3000  # Changed from 1500 to 3000 (30 seconds * 100 frames/sec)


@dataclass
class DataCollator:
    processor: Any
    max_length: int = MAX_LENGTH

    def __call__(self, features):
        feats = []

        for f in features:
            x = np.array(f["input_features"])
            if x.ndim == 3:
                x = x[0]

            t = x.shape[-1]

            # Pad or truncate to MAX_LENGTH (3000)
            if t < self.max_length:
                x = np.pad(x, ((0, 0), (0, self.max_length - t)))
            else:
                x = x[:, :self.max_length]

            feats.append(x)

        input_features = torch.tensor(
            np.stack(feats),
            dtype=torch.float32
        )

        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"] != 1, -100
        )

        decoder_input_ids = labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = self.processor.tokenizer.pad_token_id

        bos = torch.full(
            (decoder_input_ids.shape[0], 1),
            self.processor.tokenizer.bos_token_id,
            dtype=torch.long
        )

        decoder_input_ids = torch.cat([bos, decoder_input_ids[:, :-1]], dim=1)

        return {
            "input_features": input_features,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset = load_from_disk(DATASET_PATH)
    print(f"\nDataset chargé : {dataset}")

    dataset = dataset.filter(lambda ex: len(ex["labels"]) >= 3)

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language="French",
        task="transcribe"
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    model.generation_config.language = "fr"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # Freeze encoder
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    for layer in model.model.encoder.layers[-FREEZE_N:]:
        for param in layer.parameters():
            param.requires_grad = True

    for param in model.model.encoder.layer_norm.parameters():
        param.requires_grad = True

    collator = DataCollator(processor=processor)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [p.strip().lower() for p in pred_str]
        label_str = [l.strip().lower() for l in label_str]

        return {
            "wer": round(100 * wer_metric.compute(
                predictions=pred_str,
                references=label_str
            ), 2)
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,

        fp16=True,
        gradient_checkpointing=True,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,

        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,

        dataloader_num_workers=0,   # ✅ IMPORTANT WINDOWS
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        processing_class=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    print("\n===== TRAINING START =====\n")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ Modèle sauvegardé dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
