import librosa
import soundfile as sf
import os

MP3_DIR = "DATA prete"
WAV_DIR = "data/wav"

os.makedirs(WAV_DIR, exist_ok=True)

if not os.path.exists(MP3_DIR):
    raise FileNotFoundError(f"Le répertoire {MP3_DIR} n'existe pas")

converted = 0
index = 1
for file in sorted(os.listdir(MP3_DIR)):
    if file.endswith(".mp3"):
        mp3_path = os.path.join(MP3_DIR, file)
        # on normalise le nom pour correspondre à data/dataset.csv
        wav_name = f"audio_{index:03}.wav"
        wav_path = os.path.join(WAV_DIR, wav_name)

        try:
            # Charge l'audio en mono et le resample directement à 16 kHz
            samples, sr = librosa.load(mp3_path, sr=16000, mono=True)

            # Sauvegarde en WAV 16 kHz
            sf.write(wav_path, samples, 16000)
            converted += 1
            index += 1
        except Exception as e:
            print(f"❌ Erreur lors de la conversion de {file}: {e}")
            continue

print(f"✅ {converted} audios convertis en WAV 16 kHz")
