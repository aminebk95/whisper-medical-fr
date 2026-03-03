#!/usr/bin/env python3
"""
Vérification du mapping audio ↔ transcription
===============================================
Lit tous les CSV du pipeline et vérifie pour chaque entrée :
  - Le fichier audio existe
  - C'est un WAV valide (lisible)
  - La durée est dans la plage acceptable (MIN_DUR–MAX_DUR secondes)
  - La transcription n'est pas vide
  - Le préfixe source correspond au nom de fichier (pour les données réelles)

Génère un rapport détaillé et sauvegarde les problèmes dans check_report.csv.

Usage :
  python check.py
  python check.py --csv data/dataset_all_clean.csv
  python check.py --all        ← vérifie tous les CSV du pipeline
  python check.py --sample 10  ← affiche 10 exemples aléatoires OK
"""

import argparse
import csv
import os
import random
import struct
import wave
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")

# Plages de durée acceptables (secondes)
MIN_DUR = 0.5
MAX_DUR = 30.0

# CSV à vérifier par défaut avec --all
ALL_CSVS = [
    os.path.join(DATA_DIR, "dataset_all_clean.csv"),
    os.path.join(DATA_DIR, "dataset_clean_paths.csv"),
    os.path.join(DATA_DIR, "dataset_concat.csv"),
    os.path.join(DATA_DIR, "tts_generated.csv"),
]

REPORT_PATH = os.path.join(DATA_DIR, "check_report.csv")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES AUDIO
# ══════════════════════════════════════════════════════════════════════════════

def get_wav_info(path: str) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """
    Retourne (duration_s, sample_rate, error_msg).
    error_msg est None si tout va bien.
    """
    if not os.path.exists(path):
        return None, None, "fichier introuvable"
    if os.path.getsize(path) == 0:
        return None, None, "fichier vide (0 octets)"

    # Essayer wave (WAV natif)
    try:
        with wave.open(path, "r") as wf:
            frames = wf.getnframes()
            rate   = wf.getframerate()
            if rate == 0:
                return None, None, "sample_rate=0 (corrompu)"
            dur = frames / rate
            return dur, rate, None
    except wave.Error as e:
        pass
    except Exception as e:
        pass

    # Essayer librosa comme fallback (gère MP3, FLAC, etc.)
    try:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
        dur = len(y) / sr
        return dur, sr, None
    except Exception as e:
        return None, None, f"illisible : {e}"


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT CSV
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: str) -> List[Dict]:
    """
    Charge un CSV et retourne une liste de dicts.
    Accepte les formats :
      - dataset_all_clean.csv   : id, audio_path, transcription, source
      - dataset_clean_paths.csv : audio_path, transcription, duration
      - dataset_concat.csv      : audio_path, transcription
      - tts_generated.csv       : audio_path, transcription, source
    """
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception as e:
        print(f"  ❌ Impossible de lire {csv_path} : {e}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION D'UNE ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def check_row(row: Dict) -> Dict:
    """
    Vérifie une entrée CSV. Retourne un dict avec les résultats.
    """
    audio_path    = row.get("audio_path", "").strip()
    transcription = row.get("transcription", "").strip()
    source        = row.get("source", "")
    row_id        = row.get("id", "")

    result = {
        "id":            row_id,
        "audio_path":    audio_path,
        "source":        source,
        "transcription": transcription[:80],
        "status":        "OK",
        "issues":        [],
        "duration_s":    None,
        "sample_rate":   None,
    }

    # ── 1. Transcription vide ─────────────────────────────────────────────
    if not transcription:
        result["issues"].append("transcription_vide")

    # ── 2. Chemin audio absent ────────────────────────────────────────────
    if not audio_path:
        result["issues"].append("chemin_audio_absent")
        result["status"] = "ERREUR"
        return result

    # ── 3. Fichier audio existe ? ─────────────────────────────────────────
    dur, sr, err = get_wav_info(audio_path)

    if err:
        result["issues"].append(f"audio_{err.replace(' ', '_')}")
        result["status"] = "ERREUR"
        return result

    result["duration_s"]  = round(dur, 3)
    result["sample_rate"] = sr

    # ── 4. Durée hors plage ───────────────────────────────────────────────
    if dur < MIN_DUR:
        result["issues"].append(f"trop_court_{dur:.2f}s")
    elif dur > MAX_DUR:
        result["issues"].append(f"trop_long_{dur:.2f}s")

    # ── 5. Sample rate inattendu ──────────────────────────────────────────
    if sr not in (16000, 22050, 44100, 48000):
        result["issues"].append(f"sample_rate_inhabituel_{sr}Hz")

    # ── 6. Vérification du préfixe source ↔ nom de fichier ───────────────
    if source and audio_path:
        fname = Path(audio_path).stem   # ex: "cerebral_042"
        prefix_map = {
            "cerebral_pdf":       "cerebral_",
            "expression_medicale": "audio_",
            "new_rapports":       "new_",
            "tts_expression":     "tts_",
            "tts_cerebral":       "tts_",
            "tts_rapport":        "tts_",
            "concatenated":       "concat_",
            "long_original":      "",   # pas de préfixe fixe
            "short_kept":         "",
        }
        expected_prefix = prefix_map.get(source)
        if expected_prefix and not fname.startswith(expected_prefix):
            result["issues"].append(
                f"prefixe_inattendu(attendu={expected_prefix}, fichier={fname})"
            )

    # ── Statut final ──────────────────────────────────────────────────────
    if result["issues"]:
        result["status"] = "AVERTISSEMENT" if result["status"] != "ERREUR" else "ERREUR"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict], csv_name: str) -> None:
    total      = len(results)
    ok         = sum(1 for r in results if r["status"] == "OK")
    warnings   = sum(1 for r in results if r["status"] == "AVERTISSEMENT")
    errors     = sum(1 for r in results if r["status"] == "ERREUR")

    durations  = [r["duration_s"] for r in results if r["duration_s"] is not None]
    total_dur  = sum(durations)

    # Compter les types de problèmes
    all_issues: List[str] = []
    for r in results:
        all_issues.extend(r["issues"])
    issue_counts = Counter(all_issues)

    print(f"\n{'─' * 60}")
    print(f"  {csv_name}")
    print(f"{'─' * 60}")
    print(f"  Total entrées  : {total}")
    print(f"  ✅ OK          : {ok}")
    print(f"  ⚠️  Avertissements : {warnings}")
    print(f"  ❌ Erreurs     : {errors}")
    if durations:
        print(f"\n  Durée totale   : {total_dur/3600:.2f}h  ({total_dur:.0f}s)")
        print(f"  Durée moy      : {total_dur/len(durations):.1f}s")
        print(f"  Min / Max      : {min(durations):.1f}s / {max(durations):.1f}s")

    if issue_counts:
        print(f"\n  Problèmes détectés :")
        for issue, count in issue_counts.most_common():
            print(f"    {count:>5}×  {issue}")


def save_report(all_results: List[Dict], report_path: str) -> None:
    problems = [r for r in all_results if r["status"] != "OK"]
    if not problems:
        print(f"\n✅ Aucun problème — rapport non généré.")
        return
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "status", "id", "source", "duration_s", "sample_rate",
            "audio_path", "transcription", "issues"
        ])
        writer.writeheader()
        for r in problems:
            writer.writerow({**r, "issues": "; ".join(r["issues"])})
    print(f"\n📄 Rapport des problèmes → {report_path}  ({len(problems)} entrées)")


def print_samples(results: List[Dict], n: int) -> None:
    ok_results = [r for r in results if r["status"] == "OK"]
    sample = random.sample(ok_results, min(n, len(ok_results)))
    print(f"\n{'─' * 60}")
    print(f"  {n} exemples aléatoires corrects")
    print(f"{'─' * 60}")
    for r in sample:
        dur = f"{r['duration_s']:.1f}s" if r["duration_s"] else "?"
        print(f"  [{r['source'] or '?':20s}] [{dur:>6}]  {r['transcription'][:60]}")
        print(f"    ↳ {r['audio_path']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Vérifie le mapping audio ↔ transcription du pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",    default=None,
                   help="Chemin vers un CSV spécifique à vérifier.")
    p.add_argument("--all",    action="store_true",
                   help="Vérifier tous les CSV connus du pipeline.")
    p.add_argument("--sample", type=int, default=0,
                   help="Afficher N exemples aléatoires corrects.")
    p.add_argument("--no-report", action="store_true",
                   help="Ne pas sauvegarder check_report.csv.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("VÉRIFICATION DU MAPPING AUDIO ↔ TRANSCRIPTION")
    print("=" * 60)

    # Déterminer quels CSV vérifier
    if args.csv:
        csv_paths = [args.csv]
    elif args.all:
        csv_paths = [p for p in ALL_CSVS if os.path.exists(p)]
        missing   = [p for p in ALL_CSVS if not os.path.exists(p)]
        if missing:
            print(f"\nCSV absents (ignorés) :")
            for p in missing:
                print(f"  - {os.path.basename(p)}")
    else:
        # Par défaut : dataset_all_clean.csv
        default = os.path.join(DATA_DIR, "dataset_all_clean.csv")
        if not os.path.exists(default):
            print(f"\n❌ Aucun CSV trouvé. Lancez d'abord :")
            print(f"   python 08_build_full_dataset.py")
            print(f"\nOu spécifiez un CSV : python check.py --csv <chemin>")
            print(f"Ou vérifiez tout   : python check.py --all")
            return
        csv_paths = [default]

    if not csv_paths:
        print("\n❌ Aucun CSV à vérifier.")
        return

    # Vérifier chaque CSV
    all_results: List[Dict] = []

    for csv_path in csv_paths:
        csv_name = os.path.basename(csv_path)
        print(f"\nChargement : {csv_name}")
        rows = load_csv(csv_path)
        if not rows:
            print(f"  ⚠️  Vide ou illisible.")
            continue

        print(f"  {len(rows)} entrées à vérifier…")
        results = []
        for i, row in enumerate(rows):
            r = check_row(row)
            r["_csv"] = csv_name
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  … {i + 1}/{len(rows)}")

        all_results.extend(results)
        print_summary(results, csv_name)

    # Résumé global si plusieurs CSV
    if len(csv_paths) > 1:
        print(f"\n{'═' * 60}")
        print(f"  RÉSUMÉ GLOBAL — {len(csv_paths)} CSV")
        print(f"{'═' * 60}")
        total    = len(all_results)
        ok       = sum(1 for r in all_results if r["status"] == "OK")
        problems = total - ok
        durations = [r["duration_s"] for r in all_results if r["duration_s"]]
        print(f"  Total  : {total}")
        print(f"  ✅ OK  : {ok}")
        print(f"  ⚠️/❌  : {problems}")
        if durations:
            print(f"  Durée totale : {sum(durations)/3600:.2f}h")

    # Exemples aléatoires
    if args.sample > 0:
        print_samples(all_results, args.sample)

    # Sauvegarder le rapport
    if not args.no_report:
        save_report(all_results, REPORT_PATH)


if __name__ == "__main__":
    main()
