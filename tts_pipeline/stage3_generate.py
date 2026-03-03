#!/usr/bin/env python3
"""
Stage 3 — TTS Audio Generation  (Edge-TTS, French medical)
============================================================
Reads a .txt or .csv file of texts and synthesises one WAV per line
using Microsoft Edge-TTS (free, no API key, high-quality neural voices).

Output layout:
    output/generated/
        00001.wav
        00002.wav
        ...
        metadata.csv   (filename | text)

CLI usage:
    python stage3_generate.py --input texts.txt --output ./output/generated
    python stage3_generate.py --input texts.csv --output ./output/generated --voice fr-FR-HenriNeural
    python stage3_generate.py --input texts.txt --output ./output/generated --config config.yaml

Input file formats:
    .txt  → one sentence per line, blank lines skipped
    .csv  → must have a 'text' column (separator auto-detected)

French voices:
    fr-FR-DeniseNeural  — female  (default)
    fr-FR-HenriNeural   — male
    fr-FR-EloiseNeural  — female (child)
    fr-BE-CharlineNeural — Belgian French, female
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_utils import setup_logger
from utils.audio_utils import load_audio, save_audio

logger: logging.Logger = None

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "sample_rate": 22050,
    "engine":  "edge-tts",
    "voice":   "fr-FR-DeniseNeural",
    "rate":    "+0%",
    "volume":  "+0%",
    "error_log": "stage3_errors.log",
}


# ── Text loading ───────────────────────────────────────────────────────────

def load_texts(path: Path) -> List[str]:
    """
    Load lines of text from a .txt or .csv file.

    CSV: looks for a column named 'text' (case-insensitive).
    TXT: one sentence per line.
    """
    suffix = path.suffix.lower()
    texts: List[str] = []

    if suffix == ".csv":
        # Sniff separator
        sample = path.read_bytes()[:4096].decode("utf-8", errors="replace")
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        except csv.Error:
            dialect = csv.excel

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, dialect=dialect)
            col = None
            for key in (reader.fieldnames or []):
                if key.strip().lower() == "text":
                    col = key
                    break
            if col is None:
                raise ValueError(
                    f"CSV file {path.name} has no 'text' column. "
                    f"Columns found: {reader.fieldnames}"
                )
            for row in reader:
                t = row[col].strip()
                if t:
                    texts.append(t)
    else:
        # Plain text, one line per sentence
        for line in path.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t:
                texts.append(t)

    return texts


# ── Edge-TTS synthesis ─────────────────────────────────────────────────────

async def synthesise_one(
    text: str,
    voice: str,
    rate: str,
    volume: str,
    out_path: Path,
) -> None:
    """Synthesise *text* and save to *out_path* as MP3 (Edge-TTS native)."""
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts is not installed. Run: pip install edge-tts")

    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
    # Edge-TTS saves natively as MP3; we'll convert to WAV afterwards
    await communicate.save(str(out_path))


async def batch_synthesise(
    texts: List[str],
    voice: str,
    rate: str,
    volume: str,
    tmp_dir: Path,
    output_dir: Path,
    sr: int,
    error_log_path: Path,
) -> Tuple[int, int]:
    """
    Synthesise all *texts* concurrently (bounded semaphore to avoid throttling).

    Returns (n_saved, n_failed).
    """
    try:
        from tqdm.asyncio import tqdm as atqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    sem     = asyncio.Semaphore(5)  # max 5 concurrent requests
    saved   = 0
    failed  = 0
    errors: List[str] = []

    async def _generate(idx: int, text: str) -> bool:
        filename  = f"{idx:05d}.wav"
        mp3_tmp   = tmp_dir / f"{idx:05d}.mp3"
        wav_out   = output_dir / filename
        async with sem:
            try:
                await synthesise_one(text, voice, rate, volume, mp3_tmp)
                # Convert MP3 → WAV at target sample rate
                audio, _ = load_audio(mp3_tmp, target_sr=sr)
                save_audio(audio, wav_out, sr=sr)
                mp3_tmp.unlink(missing_ok=True)
                return True
            except Exception as exc:
                errors.append(f"{idx:05d} | {text[:80]} | {exc}")
                logger.warning(f"  Line {idx} failed: {exc}")
                return False

    tasks = [_generate(i + 1, t) for i, t in enumerate(texts)]

    if has_tqdm:
        results = await atqdm.gather(*tasks, desc="Generating", unit="clip")
    else:
        results = await asyncio.gather(*tasks)

    saved  = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)

    # Write error log
    if errors:
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("index | text | error\n")
            f.write("\n".join(errors) + "\n")
        logger.info(f"Error log → {error_log_path}")

    return saved, failed


# ── Metadata CSV ───────────────────────────────────────────────────────────

def write_metadata(
    texts: List[str],
    output_dir: Path,
    n_saved: int,
) -> None:
    """Write metadata.csv for successfully generated files."""
    meta_path = output_dir / "metadata.csv"
    written   = 0
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["filename", "text"])
        for i, text in enumerate(texts):
            wav = output_dir / f"{i + 1:05d}.wav"
            if wav.exists():
                writer.writerow([wav.name, text])
                written += 1
    logger.info(f"Metadata CSV written: {meta_path}  ({written} entries)")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3: TTS Audio Generation (Edge-TTS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True, help=".txt or .csv file with texts.")
    p.add_argument("--output", "-o", required=True, help="Output folder for WAV files.")
    p.add_argument("--config", "-c", default=None,  help="Path to config.yaml.")
    p.add_argument("--log-dir", default="./logs")
    p.add_argument("--voice",   default=None,
                   help="Edge-TTS voice name (overrides config).")
    p.add_argument("--rate",    default=None,
                   help="Speaking rate, e.g. '-10%%' to slow down.")
    p.add_argument("--list-voices", action="store_true",
                   help="List available French voices and exit.")
    return p.parse_args()


def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f)
        if "stage3" in full:
            cfg.update(full["stage3"])
        if "sample_rate" in full:
            cfg.setdefault("sample_rate", full["sample_rate"])
    return cfg


async def list_french_voices() -> None:
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        french = [v for v in voices if v["Locale"].startswith("fr-")]
        print("\nAvailable French Edge-TTS voices:")
        print(f"{'Name':<35} {'Gender':<8} {'Locale'}")
        print("-" * 60)
        for v in french:
            print(f"{v['ShortName']:<35} {v['Gender']:<8} {v['Locale']}")
    except ImportError:
        print("edge-tts not installed. Run: pip install edge-tts")


def main() -> None:
    global logger
    args   = parse_args()
    logger = setup_logger("stage3_generate", log_dir=args.log_dir)

    if args.list_voices:
        asyncio.run(list_french_voices())
        return

    cfg = load_config(args.config)
    if args.voice:
        cfg["voice"] = args.voice
    if args.rate:
        cfg["rate"] = args.rate

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Load texts
    try:
        texts = load_texts(input_path)
    except Exception as exc:
        logger.error(f"Failed to load texts: {exc}")
        sys.exit(1)

    if not texts:
        logger.error("No text lines found in input file.")
        sys.exit(1)

    logger.info(f"Loaded {len(texts)} text lines from {input_path.name}")
    logger.info(f"Voice : {cfg['voice']}  Rate: {cfg['rate']}  Volume: {cfg['volume']}")
    logger.info(f"Output → {output_dir}")

    tmp_dir        = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = output_dir / cfg["error_log"]

    # Run async synthesis
    n_saved, n_failed = asyncio.run(
        batch_synthesise(
            texts        = texts,
            voice        = cfg["voice"],
            rate         = cfg["rate"],
            volume       = cfg["volume"],
            tmp_dir      = tmp_dir,
            output_dir   = output_dir,
            sr           = cfg["sample_rate"],
            error_log_path = error_log_path,
        )
    )

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Write metadata.csv
    write_metadata(texts, output_dir, n_saved)

    logger.info("=" * 60)
    logger.info(f"Stage 3 complete. Generated: {n_saved} | Failed: {n_failed}")
    logger.info(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
