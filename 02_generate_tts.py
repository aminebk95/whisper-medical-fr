#!/usr/bin/env python3
"""
Script de génération TTS — Pipeline médical FR
================================================
Génère des fichiers WAV synthétiques à partir des textes existants
(RTF expression médicale, PDF cérébral, TXT rapports) via Edge-TTS.

Ordre d'exécution :
  ÉTAPE 0 : python 08_build_full_dataset.py   ← dataset réel (enregistrements)
  ÉTAPE 1 : python 02_generate_tts.py         ← CE SCRIPT (données synthétiques)
  ÉTAPE 2 : python 00_clean_audio.py
  ÉTAPE 3 : python 09_concat_audio.py
  ÉTAPE 4 : python 03_prepare_dataset.py
  ÉTAPE 5 : python 04_train_whisper.py

Installation :
  pip install edge-tts striprtf PyPDF2 librosa soundfile tqdm

Voix françaises :
  fr-FR-DeniseNeural  — féminine  (voix 1)
  fr-FR-HenriNeural   — masculine (voix 2, pour diversité des données)
"""

import asyncio
import csv
import os
import re
import shutil
from typing import List, Tuple

# ── Imports optionnels ─────────────────────────────────────────────────────
try:
    from striprtf.striprtf import rtf_to_text
    HAS_STRIPRTF = True
except ImportError:
    HAS_STRIPRTF = False

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")

# ── Chemins sources (remote desktop : C:\Users\amine&ranim\...) ────────────
RTF_PATH = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\expression médicale.rtf"
CEREBRAL_PDF = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\data\cleaned_DATASET_cérébral_texte_et_audio\textes_à_dicter_final head Mez.pdf"
NEW_RAPPORTS_DIR = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\data\new\rapports"

# ── Nouveaux fichiers texte CT / MR / US ───────────────────────────────────
CT_TXT  = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\data\textes_a_dicter_final CT.txt"
MR_TXT  = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\data\textes_a_dicter_final MR (1).txt"
US_TXT  = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\data\textes_a_dicter_final US (2).txt"

# Sortie (dans le même dossier projet)
OUTPUT_AUDIO_DIR = os.path.join(BASE, "data", "tts_generated")
OUTPUT_CSV       = os.path.join(BASE, "data", "tts_generated.csv")

# Edge-TTS — 8 voix (4 femmes / 4 hommes) avec vitesses variées
# Format : (voix, débit)  — débit varie pour diversité acoustique
VOICE_CONFIGS: List[Tuple[str, str]] = [
    ("fr-FR-DeniseNeural",               "-5%"),  # F — débit lent
    ("fr-FR-HenriNeural",                "+0%"),  # M — débit normal
    ("fr-FR-VivienneMultilingualNeural", "+5%"),  # F — débit rapide
    ("fr-FR-RemyMultilingualNeural",     "-5%"),  # M — débit lent
    ("fr-CA-SylvieNeural",               "+0%"),  # F — accent canadien
    ("fr-CA-JeanNeural",                 "+5%"),  # M — accent canadien
    ("fr-BE-CharlineNeural",             "+0%"),  # F — accent belge
    ("fr-BE-GerardNeural",               "-5%"),  # M — accent belge
]
VOLUME         = "+0%"
SAMPLE_RATE    = 16000
MIN_CHAR       = 8      # Phrases trop courtes ignorées
MAX_CONCURRENT = 5      # Requêtes Edge-TTS simultanées


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION DES TEXTES
# ══════════════════════════════════════════════════════════════════════════════

def _clean(line: str) -> str:
    return re.sub(r"^[\-\•\*]\s*", "", line.strip())


def load_rtf_sentences() -> List[Tuple[str, str]]:
    """Charge les phrases de expression médicale.rtf."""
    if not HAS_STRIPRTF or not os.path.exists(RTF_PATH):
        return []
    try:
        with open(RTF_PATH, "r", encoding="utf-8", errors="ignore") as f:
            text = rtf_to_text(f.read())
    except Exception as e:
        print(f"  ⚠️  RTF : {e}")
        return []
    return [(_clean(l), "tts_expression") for l in text.split("\n")
            if len(_clean(l)) >= MIN_CHAR]


def load_pdf_sentences() -> List[Tuple[str, str]]:
    """Charge les phrases du PDF cérébral."""
    if not HAS_PYPDF2 or not os.path.exists(CEREBRAL_PDF):
        return []
    try:
        reader   = PdfReader(CEREBRAL_PDF)
        all_text = "".join((p.extract_text() or "") + "\n" for p in reader.pages)
    except Exception as e:
        print(f"  ⚠️  PDF : {e}")
        return []
    results = []
    for line in all_text.splitlines():
        line = line.strip()
        if line and not (line.startswith("--") and "of" in line) and len(line) >= MIN_CHAR:
            results.append((line, "tts_cerebral"))
    return results


def _split_rapport(text: str) -> List[str]:
    """Découpe un rapport multi-lignes en phrases courtes."""
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= 200:
            if len(line) >= MIN_CHAR:
                sentences.append(line)
        else:
            # Ligne trop longue → découper sur ponctuation
            for part in re.split(r'(?<=[.!?])\s+', line):
                part = part.strip()
                if len(part) >= MIN_CHAR:
                    sentences.append(part)
    return sentences


def load_rapport_sentences() -> List[Tuple[str, str]]:
    """Charge les phrases de tous les fichiers TXT dans data/new/rapports/."""
    if not os.path.isdir(NEW_RAPPORTS_DIR):
        return []
    results = []
    for fname in sorted(os.listdir(NEW_RAPPORTS_DIR)):
        if not fname.lower().endswith(".txt"):
            continue
        try:
            with open(os.path.join(NEW_RAPPORTS_DIR, fname), "r",
                      encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception:
            continue
        for sentence in _split_rapport(text):
            results.append((sentence, "tts_rapport"))
    return results


def load_txt_file_sentences(path: str, source_tag: str) -> List[Tuple[str, str]]:
    """Charge les phrases d'un fichier TXT unique (CT, MR ou US).

    Stratégie :
    - Chaque bloc séparé par une ligne vide = un segment
    - Segments longs (> 200 car.) découpés sur la ponctuation
    - Phrases trop courtes ignorées
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"  ⚠️  {source_tag} : {e}")
        return []

    results = []
    # Découper sur les lignes vides → blocs/paragraphes
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Réassembler les lignes du bloc en une seule chaîne
        merged = " ".join(line.strip() for line in block.splitlines() if line.strip())
        for sentence in _split_rapport(merged):
            results.append((sentence, source_tag))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION EDGE-TTS
# ══════════════════════════════════════════════════════════════════════════════

async def _synth_one(text: str, voice: str, rate: str, mp3_path: str, wav_path: str) -> bool:
    try:
        import edge_tts
        await edge_tts.Communicate(text, voice, rate=rate, volume=VOLUME).save(mp3_path)
        if HAS_LIBROSA:
            y, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
            sf.write(wav_path, y, SAMPLE_RATE, subtype="PCM_16")
        else:
            shutil.copy(mp3_path, wav_path)
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        return True
    except Exception as e:
        print(f"  ⚠️  Échec : {text[:50]}… → {e}")
        return False


async def _generate_all(tasks: List[Tuple[str, str, str, str, str]]) -> List[bool]:
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _bounded(text, voice, rate, mp3, wav):
        async with sem:
            return await _synth_one(text, voice, rate, mp3, wav)

    coros = [_bounded(*t) for t in tasks]
    try:
        from tqdm.asyncio import tqdm as atqdm
        return list(await atqdm.gather(*coros, desc="Génération", unit="clip"))
    except ImportError:
        return list(await asyncio.gather(*coros))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("GÉNÉRATION TTS — Pipeline médical FR")
    print("=" * 60)

    # Vérifier edge-tts installé
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print("\n❌ edge-tts non installé. Lancez : pip install edge-tts")
        return

    if not HAS_LIBROSA:
        print("⚠️  librosa/soundfile absents → WAV de qualité moindre.")
        print("   Recommandé : pip install librosa soundfile\n")

    # ── 1. Charger les textes ─────────────────────────────────────────────
    print("\n[1/4] Chargement des textes…")
    rtf_items     = load_rtf_sentences()
    pdf_items     = load_pdf_sentences()
    rapport_items = load_rapport_sentences()
    ct_items      = load_txt_file_sentences(CT_TXT,  "tts_ct")
    mr_items      = load_txt_file_sentences(MR_TXT,  "tts_mr")
    us_items      = load_txt_file_sentences(US_TXT,  "tts_us")

    print(f"  RTF  (expression médicale) : {len(rtf_items):>5} phrases")
    print(f"  PDF  (cérébral)            : {len(pdf_items):>5} phrases")
    print(f"  TXT  (rapports)            : {len(rapport_items):>5} phrases")
    print(f"  TXT  (CT scanner)          : {len(ct_items):>5} phrases")
    print(f"  TXT  (MR / IRM)            : {len(mr_items):>5} phrases")
    print(f"  TXT  (US échographie)      : {len(us_items):>5} phrases")

    all_texts: List[Tuple[str, str]] = (
        rtf_items + pdf_items + rapport_items + ct_items + mr_items + us_items
    )
    print(f"  ─────────────────────────────────────")
    print(f"  Total                      : {len(all_texts):>5} phrases")

    if not all_texts:
        print("\n❌ Aucun texte trouvé. Vérifiez les chemins dans le script.")
        return

    # ── 2. Préparer les tâches ────────────────────────────────────────────
    print(f"\n[2/4] Préparation des chemins de sortie…")
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    tmp_dir = os.path.join(OUTPUT_AUDIO_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    tasks    = []
    skipped  = 0
    meta     = []   # (text, source, voice, wav_path)

    for i, (text, source) in enumerate(all_texts):
        voice, rate = VOICE_CONFIGS[i % len(VOICE_CONFIGS)]
        wav_path = os.path.join(OUTPUT_AUDIO_DIR, f"tts_{i + 1:05d}.wav")
        mp3_path = os.path.join(tmp_dir, f"tmp_{i + 1:05d}.mp3")
        meta.append((text, source, voice, wav_path))

        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
            skipped += 1
        else:
            tasks.append((text, voice, rate, mp3_path, wav_path))

    print(f"  Déjà générés (skippés) : {skipped}")
    print(f"  À générer              : {len(tasks)}")

    # ── 3. Générer ────────────────────────────────────────────────────────
    voice_names = ", ".join(v for v, _ in VOICE_CONFIGS)
    print(f"\n[3/4] Génération Edge-TTS ({len(VOICE_CONFIGS)} voix : {voice_names})…")
    if tasks:
        asyncio.run(_generate_all(tasks))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── 4. Écrire le CSV ──────────────────────────────────────────────────
    print(f"\n[4/4] Écriture du CSV : {OUTPUT_CSV}")
    written = 0
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "transcription", "source"])
        for text, source, voice, wav_path in meta:
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                writer.writerow([wav_path, text, source])
                written += 1

    failed = len(all_texts) - written - skipped

    print(f"\n{'=' * 60}")
    print(f"RÉSULTAT")
    print(f"{'=' * 60}")
    print(f"  ✅ Générés            : {written}")
    print(f"  ♻️  Déjà existants     : {skipped}")
    print(f"  ❌ Échecs             : {failed}")
    print(f"  📁 Audio             : {OUTPUT_AUDIO_DIR}")
    print(f"  📄 CSV               : {OUTPUT_CSV}")
    print(f"\nProchaine étape :")
    print(f"  python 08_build_full_dataset.py")
    print(f"  (inclut automatiquement tts_generated.csv si présent)")


if __name__ == "__main__":
    main()
