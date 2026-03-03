"""
Script de concaténation audio - Solution 1
==========================================
Problème : termes médicaux isolés (~2-3s) → Whisper hallucine
Solution  : regrouper 3-4 termes courts en phrases longues (~8-12s)

Structure attendue du CSV :
  audio_path, transcription, source

Résultat :
  - Nouveaux fichiers WAV concaténés dans OUTPUT_AUDIO_DIR
  - Nouveau CSV avec les transcriptions combinées
"""

import os
import wave
import struct
import math
import pandas as pd
import random
from pathlib import Path

# ==============================
# CONFIG
# ==============================

BASE = r"C:\Users\MSI\Downloads\DATA prete-20260203T090516Z-3-001"

CSV_INPUT = os.path.join(BASE, "data", "dataset_clean_paths.csv")  # ← sortie de 00_clean_audio_fixed.py
OUTPUT_AUDIO_DIR = os.path.join(BASE, "data", "audio_concat")             # Dossier audios concaténés
CSV_OUTPUT = os.path.join(BASE, "data", "dataset_concat.csv")       # Nouveau CSV

SAMPLE_RATE = 16000
PAUSE_MS = 400     # Pause entre termes (ms) — 400ms = naturel
MIN_TERMS = 2       # Minimum de termes à concaténer
MAX_TERMS = 4       # Maximum de termes à concaténer
TARGET_DURATION = 8.0     # Durée cible en secondes
MAX_DURATION = 28.0    # Ne pas dépasser 28s (limite Whisper = 30s)
RANDOM_SEED = 42

# Seuil pour détecter les "termes courts" à concaténer
SHORT_THRESHOLD = 4.0     # seconds — audios < 4s seront concaténés

random.seed(RANDOM_SEED)

# ==============================
# FONCTIONS UTILITAIRES
# ==============================


def get_wav_duration(path):
    """Retourne la durée d'un fichier WAV en secondes."""
    try:
        with wave.open(str(path), 'r') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def concat_wavs(paths, output_path, pause_ms=PAUSE_MS):
    """Concatène plusieurs WAV mono 16kHz avec une pause entre eux."""
    pause_samples = int(SAMPLE_RATE * pause_ms / 1000)
    pause_data = struct.pack(f'{pause_samples}h', *([0] * pause_samples))

    all_data = b''
    total_frames = 0

    for i, path in enumerate(paths):
        try:
            with wave.open(str(path), 'r') as wf:
                if wf.getframerate() != SAMPLE_RATE:
                    print(f"  ⚠️  SR incorrect ({wf.getframerate()}) : {path}")
                    return None
                if wf.getnchannels() != 1:
                    print(f"  ⚠️  Pas mono : {path}")
                    return None
                frames = wf.readframes(wf.getnframes())
                all_data += frames
                total_frames += wf.getnframes()
                if i < len(paths) - 1:
                    all_data += pause_data
                    total_frames += pause_samples
        except Exception as e:
            print(f"  ⚠️  Erreur lecture {path} : {e}")
            return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with wave.open(str(output_path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(all_data)

    return total_frames / SAMPLE_RATE


def combine_transcriptions(texts):
    """Combine plusieurs transcriptions en une seule phrase."""
    combined = []
    for t in texts:
        t = t.strip().rstrip('.')
        if t:
            combined.append(t)
    return '. '.join(combined) + '.'

# ==============================
# CHARGER CSV
# ==============================


print(f"Lecture CSV : {CSV_INPUT}")
df = pd.read_csv(CSV_INPUT)
df = df.dropna(subset=["audio_path", "transcription"])
df = df[df["transcription"].str.strip() != ""].reset_index(drop=True)
print(f"Total lignes : {len(df)}")

# ==============================
# CLASSIFIER COURT vs LONG
# ==============================

print("\nAnalyse des durées...")
df["duration"] = df["audio_path"].apply(
    lambda p: get_wav_duration(p) if str(p).endswith('.wav') else None
)

short_df = df[df["duration"].notna() & (df["duration"] < SHORT_THRESHOLD)].copy()
long_df = df[df["duration"].notna() & (df["duration"] >= SHORT_THRESHOLD)].copy()
bad_df = df[df["duration"].isna()].copy()  # MP3 ou erreur

print(f"  Audios courts (< {SHORT_THRESHOLD}s) : {len(short_df)}")
print(f"  Audios longs  (≥ {SHORT_THRESHOLD}s) : {len(long_df)}")
print(f"  Audios non-WAV / erreur        : {len(bad_df)}")

# ==============================
# CONCATÉNATION DES COURTS
# ==============================

print(f"\nConcaténation des termes courts (groupes de {MIN_TERMS}-{MAX_TERMS})...")
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

# Mélanger pour varier les combinaisons
short_list = short_df.to_dict('records')
random.shuffle(short_list)

concat_rows = []
i = 0
group_idx = 0

while i < len(short_list):
    group = []
    group_dur = 0.0
    group_texts = []
    group_paths = []

    # Remplir le groupe jusqu'à TARGET_DURATION ou MAX_TERMS
    while i < len(short_list) and len(group) < MAX_TERMS:
        row = short_list[i]
        dur = row.get("duration", 0) or 0
        pause_dur = PAUSE_MS / 1000

        if group_dur + dur + pause_dur > MAX_DURATION:
            break

        group.append(row)
        group_dur += dur + pause_dur
        group_texts.append(str(row["transcription"]))
        group_paths.append(row["audio_path"])
        i += 1

    # Ignorer les groupes trop petits
    if len(group) < MIN_TERMS:
        # Garder quand même le terme seul s'il est suffisamment long
        if len(group) == 1 and group[0].get("duration", 0) >= 1.5:
            concat_rows.append({
                "audio_path": group[0]["audio_path"],
                "transcription": group[0]["transcription"],
                "source": "short_kept",
                "duration": group[0].get("duration"),
                "n_terms": 1,
            })
        i += 1
        continue

    # Concaténer
    out_name = f"concat_{group_idx:04d}.wav"
    out_path = os.path.join(OUTPUT_AUDIO_DIR, out_name)
    final_dur = concat_wavs(group_paths, out_path, pause_ms=PAUSE_MS)

    if final_dur is None:
        print(f"  ⚠️  Échec groupe {group_idx}")
        continue

    transcription = combine_transcriptions(group_texts)
    concat_rows.append({
        "audio_path": out_path,
        "transcription": transcription,
        "source": "concatenated",
        "duration": final_dur,
        "n_terms": len(group),
    })
    group_idx += 1

print(f"  Groupes créés      : {group_idx}")
print(f"  Fichiers générés   : {len([r for r in concat_rows if r['source']=='concatenated'])}")

# ==============================
# ASSEMBLER LE DATASET FINAL
# ==============================

# Garder les longs tels quels
long_rows = long_df.assign(source="long_original", n_terms=1).to_dict('records')

# Combiner
all_rows = long_rows + concat_rows
final_df = pd.DataFrame(all_rows)
final_df = final_df[["audio_path", "transcription", "source", "duration", "n_terms"]]
final_df = final_df.dropna(subset=["audio_path", "transcription"])
final_df = final_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# ==============================
# STATISTIQUES FINALES
# ==============================

print(f"\n=== DATASET FINAL ===")
print(f"Total exemples     : {len(final_df)}")
print(f"  long_original    : {(final_df['source']=='long_original').sum()}")
print(f"  concatenated     : {(final_df['source']=='concatenated').sum()}")
print(f"  short_kept       : {(final_df['source']=='short_kept').sum()}")
print(
    f"\nDurée moy/min/max  : {final_df['duration'].mean():.1f}s / {final_df['duration'].min():.1f}s / {final_df['duration'].max():.1f}s")
print(
    f"Mots/transcription : {final_df['transcription'].apply(lambda x: len(str(x).split())).mean():.1f} mots en moyenne")

# Aperçu
print(f"\nExemples concaténés :")
sample = final_df[final_df['source'] == 'concatenated'].head(3)
for _, row in sample.iterrows():
    print(f"  [{row['n_terms']} termes, {row['duration']:.1f}s] {str(row['transcription'])[:80]}...")

# ==============================
# SAUVEGARDER
# ==============================

os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
final_df[["audio_path", "transcription"]].to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
print(f"\n✅ CSV sauvegardé : {CSV_OUTPUT}")
print(f"✅ Audios dans    : {OUTPUT_AUDIO_DIR}")
print(f"\nProchaine étape :")
print(f"  Mettre à jour CSV_PATH dans 03_prepare_dataset_v2.py :")
print(f'  CSV_PATH = "{CSV_OUTPUT}"')
print(f"  Puis : python 03_prepare_dataset_v2.py")
print(f"  Puis : python 04_train_whisper_v8.py")
