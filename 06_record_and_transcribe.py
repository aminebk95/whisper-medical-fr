#!/usr/bin/env python3
"""
06_record_and_transcribe.py
============================
Enregistrement microphone en temps réel + transcription Whisper.

Raccourcis clavier :
  F1 → Démarrer l'enregistrement
  F2 → Pause / Reprendre
  F3 → Terminer et transcrire

Installation (si manquant) :
  pip install sounddevice keyboard soundfile numpy
"""

import os
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import keyboard
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE       = os.path.dirname(os.path.abspath(__file__))
# Charge depuis HuggingFace Hub si le dossier local n'existe pas
_LOCAL     = os.path.join(BASE, "data", "whisper-medical-fr-v6")
MODEL_DIR  = _LOCAL if os.path.isdir(_LOCAL) else "amnbk/whisper-medical-fr"
OUTPUT_WAV = os.path.join(BASE, "recorded_temp.wav")

SAMPLE_RATE = 16000   # Hz — format attendu par Whisper

KEY_START = "f1"   # Démarrer
KEY_PAUSE = "f2"   # Pause / Reprendre
KEY_END   = "f3"   # Terminer


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAT DE L'ENREGISTREMENT
# ══════════════════════════════════════════════════════════════════════════════

class RecordState:
    def __init__(self):
        self.recording = False
        self.paused    = False
        self.done      = False
        self.chunks    = []

state = RecordState()


def on_start(_=None):
    if not state.recording and not state.done:
        state.recording = True
        state.paused    = False
        print("\n  [F1] Enregistrement DÉMARRÉ  — F2=pause  F3=terminer")


def on_pause(_=None):
    if state.recording:
        state.paused = not state.paused
        if state.paused:
            print("\n  [F2] PAUSE — appuyez à nouveau sur F2 pour reprendre")
        else:
            print("\n  [F2] REPRISE de l'enregistrement...")


def on_end(_=None):
    if state.recording:
        state.recording = False
        state.done      = True
        print("\n  [F3] Enregistrement TERMINÉ")


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK AUDIO (thread sounddevice)
# ══════════════════════════════════════════════════════════════════════════════

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"  [warn] {status}", file=sys.stderr)
    if state.recording and not state.paused:
        state.chunks.append(indata.copy())


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT MODÈLE
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    print(f"  Modèle      : {MODEL_DIR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Appareil    : {device}")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")

    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    decoder_prompt_ids = processor.get_decoder_prompt_ids(
        language="fr", task="transcribe"
    )
    model.generation_config.forced_decoder_ids = decoder_prompt_ids
    return processor, model, device


# ══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

def transcribe(audio_array: np.ndarray, processor, model, device) -> str:
    inputs = processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)[0]
    return processor.tokenizer.decode(predicted_ids, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ENREGISTREMENT + TRANSCRIPTION — Whisper médical FR")
    print("=" * 60)
    print()
    print("  F1  →  Démarrer l'enregistrement")
    print("  F2  →  Pause / Reprendre")
    print("  F3  →  Terminer et transcrire")
    print()

    # Charger le modèle en premier (peut prendre quelques secondes)
    print("Chargement du modèle Whisper...")
    processor, model, device = load_model()
    print("Modèle prêt.\n")

    # Brancher les touches
    keyboard.on_press_key(KEY_START, on_start)
    keyboard.on_press_key(KEY_PAUSE, on_pause)
    keyboard.on_press_key(KEY_END,   on_end)

    print(f"En attente — appuyez sur F1 pour commencer à parler.\n")

    # Ouvrir le flux microphone
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        while not state.done:
            time.sleep(0.05)

    keyboard.unhook_all()

    # Vérifier qu'on a du son
    if not state.chunks:
        print("\nAucun audio enregistré.")
        return

    # Assembler les morceaux
    audio = np.concatenate(state.chunks, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE
    print(f"\nDurée enregistrée : {duration:.1f}s")

    # Sauvegarder le WAV
    os.makedirs(os.path.dirname(OUTPUT_WAV), exist_ok=True)
    sf.write(OUTPUT_WAV, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"Audio sauvegardé  : {OUTPUT_WAV}")

    # Transcrire
    print("\nTranscription en cours...")
    text = transcribe(audio, processor, model, device)

    print()
    print("=" * 60)
    print("TRANSCRIPTION :")
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
