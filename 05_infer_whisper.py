import argparse
import os
from typing import List

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_model(model_dir: str):
    """
    Charge le modèle fine-tuné et le processor.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Le dossier modèle '{model_dir}' est introuvable. "
            f"Vérifiez que 04_train_whisper.py a bien été exécuté."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Appareil utilisé : {device}")

    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    # Forcer la génération en français (comme à l'entraînement)
    decoder_prompt_ids = processor.get_decoder_prompt_ids(
        language="fr",
        task="transcribe",
    )
    model.generation_config.forced_decoder_ids = decoder_prompt_ids

    return processor, model, device


def list_audio_files(audio_dir: str) -> List[str]:
    """
    Liste les fichiers audio supportés dans un dossier.
    """
    supported_ext = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    files = []
    for name in sorted(os.listdir(audio_dir)):
        if name.lower().endswith(supported_ext):
            files.append(os.path.join(audio_dir, name))
    return files


def transcribe_file(
    audio_path: str,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
) -> str:
    """
    Transcrit un fichier audio unique avec le modèle Whisper fine-tuné.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

    # Chargement audio en mono 16 kHz (format attendu par Whisper)
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)[0]

    transcription = processor.tokenizer.decode(
        predicted_ids,
        skip_special_tokens=True,
    )

    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Tester le modèle Whisper fine-tuné sur des fichiers audio."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model/whisper-base-medical-fr-no-aug-frozen",
        help="Chemin vers le dossier du modèle fine-tuné (save_pretrained).",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="*",
        help="Un ou plusieurs chemins de fichiers audio à transcrire.",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        help="Dossier contenant des fichiers audio à transcrire.",
    )

    args = parser.parse_args()

    if not args.audio and not args.audio_dir:
        raise SystemExit(
            "Vous devez fournir soit --audio <fichiers>, soit --audio_dir <dossier>."
        )

    processor, model, device = load_model(args.model_dir)

    audio_files: List[str] = []

    if args.audio:
        audio_files.extend(args.audio)

    if args.audio_dir:
        audio_files.extend(list_audio_files(args.audio_dir))

    if not audio_files:
        raise SystemExit("Aucun fichier audio trouvé à transcrire.")

    print(f"Nombre de fichiers à transcrire : {len(audio_files)}")

    for path in audio_files:
        try:
            text = transcribe_file(path, processor, model, device)
            print("=" * 80)
            print(f"Fichier : {path}")
            print("Transcription :")
            print(text)
        except Exception as e:
            print("=" * 80)
            print(f"Erreur sur le fichier {path} : {e}")


if __name__ == "__main__":
    main()

