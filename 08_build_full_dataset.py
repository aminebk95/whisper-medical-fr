import os
import csv
import re
from typing import List, Tuple

from striprtf.striprtf import rtf_to_text

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None  # type: ignore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- PATHS (remote desktop : C:\Users\amine&ranim\Desktop\DATA prete-...) ---
DATA_DIR = os.path.join(BASE_DIR, "data")

# 1) Expression médicale : RTF + WAV
RTF_PATH     = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\expression médicale.rtf"
EXPR_WAV_DIR = os.path.join(DATA_DIR, "wav")

# 2) Dataset cérébral (PDF + WAV)
CEREBRAL_DIR     = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\data\cleaned_DATASET_cérébral_texte_et_audio"
CEREBRAL_PDF     = os.path.join(CEREBRAL_DIR, "textes_à_dicter_final head Mez.pdf")
CEREBRAL_WAV_DIR = os.path.join(CEREBRAL_DIR, "wav")

# 3) Nouveau dataset (audio wav + rapports txt)
NEW_AUDIO_DIR    = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\data\new\audio"
NEW_RAPPORTS_DIR = r"C:\Users\amine&ranim\Desktop\DATA prete-20260203T090516Z-3-001\data\new\rapports"

# 4) Fichier de sortie global
# Version "clean" : uniquement WAV, aucune donnée augmentée, pas d'ancien CSV utilisé.
OUTPUT_CSV = os.path.join(DATA_DIR, "dataset_all_clean.csv")


def clean_sentence(line: str) -> str:
    """Nettoie une ligne de texte (similaire à 02_rtf_to_csv.py)."""
    line = line.strip()
    line = re.sub(r"^[\-\•\*]\s*", "", line)
    return line


def build_expression_rows() -> List[Tuple[str, str, str, str]]:
    """
    Construit les paires (id, audio_path, transcription, source) pour
    l'expression médicale directement à partir du RTF + WAV:
    - Texte: 'expression médicale.rtf'
    - Audio: 'data/wav/audio_XXX.wav'
    """
    rows: List[Tuple[str, str, str, str]] = []

    if not os.path.exists(RTF_PATH):
        return rows

    try:
        with open(RTF_PATH, "r", encoding="utf-8", errors="ignore") as f:
            text = rtf_to_text(f.read())
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la lecture du RTF: {e}")

    sentences = [
        clean_sentence(line)
        for line in text.split("\n")
        if len(clean_sentence(line)) > 5
    ]

    for i, sentence in enumerate(sentences, start=1):
        wav_name = f"audio_{i:03}.wav"
        audio_path = os.path.join(EXPR_WAV_DIR, wav_name)

        if not os.path.exists(audio_path):
            # On ignore les phrases qui n'ont pas de WAV correspondant
            continue

        row_id = f"expr_{i:03}"
        rows.append((row_id, audio_path, sentence, "expression_medicale"))

    return rows


def extract_cerebral_pairs_from_pdf() -> List[Tuple[int, str]]:
    """
    Extrait (index, phrase) à partir du PDF cérébral.

    Version robuste: on ne dépend PAS d'un numéro en fin de ligne.
    - On récupère tout le texte du PDF.
    - On découpe par lignes.
    - On garde les lignes non vides, suffisamment longues, et qui ne ressemblent
      pas à des en-têtes de page.
    - On renvoie (index_sequentiel, phrase).
    """
    if PdfReader is None:
        raise ImportError(
            "PyPDF2 n'est pas installé. Installez-le avec 'pip install PyPDF2' pour traiter le PDF cérébral."
        )

    if not os.path.exists(CEREBRAL_PDF):
        raise FileNotFoundError(f"PDF non trouvé : {CEREBRAL_PDF}")

    reader = PdfReader(CEREBRAL_PDF)
    all_text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = ""
        if page_text:
            all_text += page_text + "\n"

    pairs: List[Tuple[int, str]] = []
    idx = 1
    for raw_line in all_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # On saute les séparateurs/headers de pages très courts ou techniques
        if line.startswith("--") and "of" in line:
            continue
        if len(line) < 5:
            continue

        sentence = line.strip()
        pairs.append((idx, sentence))
        idx += 1

    return pairs


def build_cerebral_rows() -> List[Tuple[str, str, str, str]]:
    """
    Associe le texte du PDF à chaque WAV cérébral.

    Stratégie pour être robuste aux différences de numérotation:
    - On lit toutes les phrases pertinentes du PDF (via extract_cerebral_pairs_from_pdf)
      et on garde uniquement la partie texte.
    - On liste tous les fichiers .wav du dossier '.../DATASET cérébral texte et audio/wav'
      et on les trie.
    - On associe ensuite dans l'ordre: 1ère phrase -> 1er WAV, etc.
      (limité au min(nb_phrases, nb_wav)).
    """
    rows: List[Tuple[str, str, str, str]] = []

    if not os.path.isdir(CEREBRAL_WAV_DIR):
        return rows

    # 1) Récupérer les phrases depuis le PDF
    pairs = extract_cerebral_pairs_from_pdf()
    sentences = [sentence for _, sentence in pairs]

    # 2) Lister tous les WAV disponibles
    wav_files = [
        f
        for f in os.listdir(CEREBRAL_WAV_DIR)
        if f.lower().endswith(".wav")
    ]
    wav_files.sort()

    # 3) Associer dans l'ordre
    for i, (filename, sentence) in enumerate(
        zip(wav_files, sentences), start=1
    ):
        audio_path = os.path.join(CEREBRAL_WAV_DIR, filename)
        row_id = f"cerebral_{i:03}"
        rows.append((row_id, audio_path, sentence, "cerebral_pdf"))

    return rows


def build_new_rows() -> List[Tuple[str, str, str, str]]:
    """
    Associe chaque rapport txt de 'data/new/rapports' à son audio wav
    dans 'data/new/audio'. Hypothèse : même nom de base (ex: '27.wav' et '27.txt').
    """
    rows: List[Tuple[str, str, str, str]] = []

    if not (os.path.isdir(NEW_AUDIO_DIR) and os.path.isdir(NEW_RAPPORTS_DIR)):
        return rows

    rapport_files = [
        f for f in os.listdir(NEW_RAPPORTS_DIR) if f.lower().endswith(".txt")
    ]

    for rapport in rapport_files:
        base, _ = os.path.splitext(rapport)
        audio_candidate = f"{base}.wav"
        audio_path = os.path.join(NEW_AUDIO_DIR, audio_candidate)

        if not os.path.exists(audio_path):
            # On essaie quelques variantes simples (zéro à gauche, etc.) si besoin
            # mais principalement on s'en tient à base + '.wav'
            continue

        rapport_path = os.path.join(NEW_RAPPORTS_DIR, rapport)
        try:
            with open(rapport_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception:
            continue

        if not text:
            continue

        row_id = f"new_{base}"
        rows.append((row_id, audio_path, text, "new_rapports"))

    return rows


def build_tts_rows() -> List[Tuple[str, str, str, str]]:
    """
    Charge les données TTS générées par 02_generate_tts.py.
    Retourne [] si tts_generated.csv n'existe pas encore.
    """
    tts_csv = os.path.join(DATA_DIR, "tts_generated.csv")
    if not os.path.exists(tts_csv):
        return []

    rows: List[Tuple[str, str, str, str]] = []
    with open(tts_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            audio_path    = row.get("audio_path", "").strip()
            transcription = row.get("transcription", "").strip()
            source        = row.get("source", "tts_generated").strip()
            if audio_path and transcription and os.path.exists(audio_path):
                rows.append((f"tts_{i:05d}", audio_path, transcription, source))
    return rows


def main() -> None:
    from collections import Counter

    all_rows: List[Tuple[str, str, str, str]] = []

    # 1) Expression médicale (RTF + WAV)
    expr_rows = build_expression_rows()
    all_rows.extend(expr_rows)

    # 2) Dataset cérébral (PDF + WAV)
    try:
        cerebral_rows = build_cerebral_rows()
        all_rows.extend(cerebral_rows)
    except Exception as e:
        cerebral_rows = []
        print(f"⚠️ Erreur lors du traitement du dataset cérébral : {e}")

    # 3) Nouveau dataset (audio + rapports)
    new_rows = build_new_rows()
    all_rows.extend(new_rows)

    # 4) Données TTS générées (02_generate_tts.py) — optionnel
    tts_rows = build_tts_rows()
    all_rows.extend(tts_rows)

    if not all_rows:
        raise RuntimeError("Aucune donnée trouvée pour construire dataset_all.csv.")

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "audio_path", "transcription", "source"])
        for row in all_rows:
            writer.writerow(row)

    counts = Counter(src for _, _, _, src in all_rows)
    print(f"Dataset global créé : {OUTPUT_CSV}")
    print(f"  Total d'entrées : {len(all_rows)}")
    for src, n in counts.items():
        print(f"  {src}: {n}")
    if tts_rows:
        print(f"\n  ℹ️  Données TTS incluses depuis tts_generated.csv")
    else:
        print(f"\n  ℹ️  Pas de données TTS (lancez d'abord 02_generate_tts.py)")


if __name__ == "__main__":
    main()

