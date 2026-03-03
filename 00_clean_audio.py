"""
Script de nettoyage audio - Pipeline médical FR
================================================
Ordre correct :
  ÉTAPE 1 : python 00_clean_audio.py        ← CE SCRIPT
  ÉTAPE 2 : python 09_concat_audio.py
  ÉTAPE 3 : python 03_prepare_dataset_v2.py
  ÉTAPE 4 : python 04_train_whisper_v8.py

Installation recommandée avant de lancer :
  pip install librosa soundfile noisereduce
"""

import os
import wave
import struct
import math
import pandas as pd
from pathlib import Path

# ==============================
# CONFIG — chemins corrigés (raw strings)
# ==============================

BASE     = r"C:\Users\MSI\Downloads\DATA prete-20260203T090516Z-3-001"

CSV_INPUT    = os.path.join(BASE, "data", "dataset_all_clean.csv")
OUTPUT_DIR   = os.path.join(BASE, "data", "audio_clean")
CSV_OUTPUT   = os.path.join(BASE, "data", "dataset_clean_paths.csv")

SAMPLE_RATE  = 16000
MIN_DURATION = 0.5
TARGET_RMS   = 5000
TOP_DB       = 20

# ==============================
# DÉTECTION LIBRAIRIES
# ==============================

try:
    import librosa
    import numpy as np
    import soundfile as sf
    HAS_LIBROSA = True
    print("✅ librosa détecté — nettoyage complet (WAV + MP3)")
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  librosa absent — nettoyage basique WAV uniquement")
    print("   Recommandé : pip install librosa soundfile noisereduce\n")

try:
    import noisereduce as nr
    HAS_NR = True
    print("✅ noisereduce détecté — suppression bruit activée")
except ImportError:
    HAS_NR = False
    if HAS_LIBROSA:
        print("⚠️  noisereduce absent — sans suppression bruit")

print()

# ==============================
# NETTOYAGE COMPLET (avec librosa)
# ==============================

if HAS_LIBROSA:
    import numpy as np
    import soundfile as sf

    def clean_audio(input_path, output_path):
        y, sr = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)
        if len(y) == 0:
            return None, "audio vide"

        if HAS_NR:
            noise_len = min(int(0.3 * sr), len(y) // 4)
            if noise_len > 100:
                y = nr.reduce_noise(
                    y=y, sr=sr,
                    y_noise=y[:noise_len],
                    prop_decrease=0.75,
                    stationary=True
                )

        for db in [TOP_DB, 15, 10]:
            y_trim, _ = librosa.effects.trim(y, top_db=db)
            if len(y_trim) >= int(MIN_DURATION * sr):
                break
        else:
            return None, f"trop court après trim ({len(y_trim)/sr:.2f}s)"

        rms = np.sqrt(np.mean(y_trim ** 2))
        if rms <= 0:
            return None, "RMS nul"
        scale  = (TARGET_RMS / 32768) / rms
        y_norm = np.clip(y_trim * scale, -1.0, 1.0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y_norm, SAMPLE_RATE, subtype='PCM_16')
        return len(y_norm) / SAMPLE_RATE, None

# ==============================
# NETTOYAGE BASIQUE (sans librosa)
# ==============================

else:
    def clean_audio(input_path, output_path):
        if not str(input_path).lower().endswith('.wav'):
            return None, "MP3 non supporté sans librosa"
        try:
            with wave.open(str(input_path), 'r') as wf:
                sr       = wf.getframerate()
                n_ch     = wf.getnchannels()
                sw       = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw      = wf.readframes(n_frames)
        except Exception as e:
            return None, str(e)

        if sw != 2:
            return None, "format non supporté (pas 16 bits)"

        samples = list(struct.unpack(f'{n_frames * n_ch}h', raw))
        if n_ch == 2:
            samples = [(samples[i]+samples[i+1])//2 for i in range(0, len(samples), 2)]

        if sr != SAMPLE_RATE:
            ratio   = SAMPLE_RATE / sr
            new_len = int(len(samples) * ratio)
            resampled = []
            for k in range(new_len):
                idx  = k / ratio
                lo   = int(idx)
                hi   = min(lo + 1, len(samples) - 1)
                frac = idx - lo
                resampled.append(int(samples[lo] * (1-frac) + samples[hi] * frac))
            samples = resampled

        max_amp   = max(abs(s) for s in samples) if samples else 1
        threshold = max_amp * (10 ** (-TOP_DB / 20))
        pad       = int(0.05 * SAMPLE_RATE)

        start = next((j for j, s in enumerate(samples) if abs(s) > threshold), 0)
        end   = next((j for j in range(len(samples)-1, -1, -1) if abs(samples[j]) > threshold), len(samples)-1)
        start = max(0, start - pad)
        end   = min(len(samples), end + pad)
        samples = samples[start:end]

        if len(samples) < int(MIN_DURATION * SAMPLE_RATE):
            return None, f"trop court ({len(samples)/SAMPLE_RATE:.2f}s)"

        rms   = math.sqrt(sum(s*s for s in samples) / len(samples))
        if rms <= 0:
            return None, "RMS nul"
        scale   = min(TARGET_RMS / rms, 32767 / max(abs(s) for s in samples))
        samples = [max(-32768, min(32767, int(s * scale))) for s in samples]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with wave.open(output_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(struct.pack(f'{len(samples)}h', *samples))

        return len(samples) / SAMPLE_RATE, None

# ==============================
# TRAITEMENT
# ==============================

print(f"Lecture CSV : {CSV_INPUT}")
df = pd.read_csv(CSV_INPUT)
df = df.dropna(subset=["audio_path", "transcription"])
df = df[df["transcription"].str.strip() != ""].reset_index(drop=True)
print(f"Fichiers à nettoyer : {len(df)}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

results   = []
processed = 0
skipped   = 0

for idx, row in df.iterrows():
    input_path    = str(row["audio_path"])
    transcription = str(row["transcription"]).strip()

    stem        = Path(input_path).stem
    output_path = os.path.join(OUTPUT_DIR, f"{stem}.wav")
    if os.path.exists(output_path):
        output_path = os.path.join(OUTPUT_DIR, f"{stem}_{idx}.wav")

    duration, error = clean_audio(input_path, output_path)

    if error:
        skipped += 1
        if skipped <= 10:
            print(f"  [SKIP] {Path(input_path).name} : {error}")
        continue

    results.append({
        "audio_path":    output_path,
        "transcription": transcription,
        "duration":      round(duration, 2),
    })
    processed += 1

    if processed % 100 == 0:
        print(f"  Traités : {processed}/{len(df)}")

# ==============================
# RAPPORT FINAL
# ==============================

print(f"\n{'='*50}")
print(f"RÉSULTAT NETTOYAGE")
print(f"{'='*50}")
print(f"  ✅ Nettoyés  : {processed}")
print(f"  ⚠️  Ignorés  : {skipped}")
if processed + skipped > 0:
    print(f"  Taux succès  : {processed/(processed+skipped)*100:.1f}%")

if results:
    durations = [r["duration"] for r in results]
    print(f"\n  Durée moy : {sum(durations)/len(durations):.1f}s")
    print(f"  Durée min : {min(durations):.1f}s")
    print(f"  Durée max : {max(durations):.1f}s")
    print(f"\n  Courts (< 4s) : {sum(1 for d in durations if d < 4)}")
    print(f"  Longs  (≥ 4s) : {sum(1 for d in durations if d >= 4)}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")

    print(f"\n✅ Audios nettoyés : {OUTPUT_DIR}")
    print(f"✅ CSV sauvegardé  : {CSV_OUTPUT}")
    print(f"\n{'='*50}")
    print(f"ÉTAPE SUIVANTE :")
    print(f"  Dans 09_concat_audio.py, mettre :")
    print(f'  CSV_INPUT = r"{CSV_OUTPUT}"')
    print(f"  Puis : python 09_concat_audio.py")