import os
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du device: {device}")

MODEL_PATH = "model/whisper-medical-fr"
TEST_AUDIO = "data/wav/audio_001.wav"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le modèle {MODEL_PATH} n'existe pas. Exécutez d'abord 04_train_whisper.py")

try:
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH
    ).to(device)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")

try:
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du processeur: {e}")

def transcribe(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Le fichier audio {audio_path} n'existe pas")
    
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        raise IOError(f"Erreur lors du chargement de l'audio: {e}")

    try:
        inputs = processor(
            audio,
            sampling_rate=16000,
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

if __name__ == "__main__":
    if os.path.exists(TEST_AUDIO):
        result = transcribe(TEST_AUDIO)
        print(f"Transcription: {result}")
    else:
        print(f"⚠️  Le fichier {TEST_AUDIO} n'existe pas")
        print("Utilisation: transcribe('chemin/vers/audio.wav')")
