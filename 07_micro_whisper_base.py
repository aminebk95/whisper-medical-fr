import os
import torch
import librosa
import sounddevice as sd
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Utilisation du GPU si disponible, sinon CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du device: {device}")

MODEL_NAME = "openai/whisper-small"
OUTPUT_AUDIO = "data/wav/micro_base_test.wav"
SAMPLING_RATE = 16000
DURATION_SECONDS = 10  # durée d'enregistrement en secondes (modifiable)

# Charger le processeur (français, transcription)
try:
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language="fr",
        task="transcribe",
    )
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du processeur: {e}")

# Charger le modèle de base
try:
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME
    ).to(device)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")


def transcribe(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Le fichier audio {audio_path} n'existe pas")

    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
    except Exception as e:
        raise IOError(f"Erreur lors du chargement de l'audio: {e}")

    try:
        inputs = processor(
            audio,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt"
        ).input_features.to(device)

        # Forcer la langue française et la tâche de transcription
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="fr",
            task="transcribe"
        )

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=256
            )

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        return transcription
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la transcription: {e}")


def record_from_microphone(output_path, duration_seconds=DURATION_SECONDS):
    """Enregistre depuis le micro et sauvegarde en WAV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Enregistrement pendant {duration_seconds} secondes...")
    print("Parle maintenant...")
    recording = sd.rec(
        int(duration_seconds * SAMPLING_RATE),
        samplerate=SAMPLING_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Enregistrement terminé, sauvegarde du fichier...")

    sf.write(output_path, recording, SAMPLING_RATE)
    print(f"Audio sauvegardé dans: {output_path}")


if __name__ == "__main__":
    # 1) Enregistrer ta voix
    record_from_microphone(OUTPUT_AUDIO, DURATION_SECONDS)

    # 2) Transcrire ce que tu viens de dire
    print("Transcription (modèle de base) en cours...")
    result = transcribe(OUTPUT_AUDIO)
    print("\n=== TRANSCRIPTION (WHISPER SMALL BASE) ===")
    print(result)
    print("==========================================")

