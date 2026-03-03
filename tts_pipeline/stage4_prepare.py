#!/usr/bin/env python3
"""
Stage 4 — Dataset Preparation
===============================
Validates, transcribes, normalises, and splits an audio corpus into
the LJSpeech format used by Coqui TTS / VITS / XTTS.

Input layout (two supported modes):
  A) Folder of WAVs + a metadata.csv  (filename | text)        ← Stage 3 output
  B) Folder of WAVs alone             (Whisper transcribes all) ← Stage 1 output

Output layout:
  output/dataset/
      wavs/           ← symlinked or copied WAV files
      metadata.csv    ← filename|text|normalized_text  (no header, pipe-sep)
      train.txt
      val.txt
      test.txt
      stats.csv       ← per-file duration, RMS, SNR, WER, flagged
      rejected.txt    ← files excluded from the dataset

CLI usage:
    python stage4_prepare.py --input ./output/generated --output ./output/dataset
    python stage4_prepare.py --input ./output/cleaned   --output ./output/dataset --no-metadata
    python stage4_prepare.py --input ./output/generated --output ./output/dataset --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_utils import setup_logger
from utils.audio_utils import load_audio, get_duration, compute_rms_db, compute_snr
from utils.text_utils import normalize_text

logger: logging.Logger = None

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "sample_rate": 22050,
    "whisper_model": "medium",
    "whisper_language": "fr",
    "whisper_device": "cuda",
    "mismatch_threshold": 0.35,
    "text_normalization": {
        "lowercase": True,
        "remove_special_chars": True,
        "expand_abbreviations": True,
    },
    "min_duration_s": 1.0,
    "max_duration_s": 15.0,
    "train_ratio": 0.90,
    "val_ratio":   0.05,
    "test_ratio":  0.05,
    "random_seed": 42,
}


# ── Metadata loading ───────────────────────────────────────────────────────

def load_metadata(meta_path: Path) -> Dict[str, str]:
    """
    Load a metadata CSV into {filename_stem: text}.

    Accepts:
      - pipe-separated (LJSpeech style): filename|text  or  filename|text|norm
      - comma/tab CSV with a 'filename' and 'text' column
    """
    data: Dict[str, str] = {}
    if not meta_path.exists():
        return data

    raw = meta_path.read_text(encoding="utf-8")
    sample = raw[:2048]

    # Detect delimiter
    if "|" in sample:
        delim = "|"
    else:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delim = dialect.delimiter
        except csv.Error:
            delim = ","

    reader = csv.reader(raw.splitlines(), delimiter=delim)
    rows   = [r for r in reader if r]

    # Detect if first row is a header
    first = [c.strip().lower() for c in rows[0]] if rows else []
    if "filename" in first or "text" in first:
        rows = rows[1:]   # skip header

    for row in rows:
        if len(row) < 2:
            continue
        filename = Path(row[0].strip()).stem   # strip extension + path
        text     = row[1].strip()
        if filename and text:
            data[filename] = text

    return data


# ── Whisper transcription ──────────────────────────────────────────────────

class WhisperTranscriber:
    def __init__(self, model_size: str, language: str, device: str):
        self._model_size = model_size
        self._language   = language
        self._device     = device
        self._model      = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import whisper
            import torch
            device = self._device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available — using CPU for Whisper.")
                device = "cpu"
            logger.info(f"Loading Whisper '{self._model_size}' on {device}…")
            self._model = whisper.load_model(self._model_size, device=device)
        except ImportError:
            raise RuntimeError(
                "openai-whisper is not installed. Run: pip install openai-whisper"
            )

    def transcribe(self, wav_path: Path) -> str:
        self._load()
        result = self._model.transcribe(
            str(wav_path),
            language=self._language,
            fp16=(self._device == "cuda"),
        )
        return result["text"].strip()


# ── WER calculation ────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between two strings."""
    try:
        from jiwer import wer
        return float(wer(reference.lower(), hypothesis.lower()))
    except ImportError:
        # Fallback: simple word-level edit distance ratio
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        # DP edit distance
        dp = list(range(len(hyp_words) + 1))
        for r in ref_words:
            new_dp = [dp[0] + 1] + [0] * len(hyp_words)
            for j, h in enumerate(hyp_words):
                new_dp[j + 1] = min(
                    dp[j] + (0 if r == h else 1),
                    new_dp[j] + 1,
                    dp[j + 1] + 1,
                )
            dp = new_dp
        return dp[-1] / max(len(ref_words), 1)


# ── Dataset building ───────────────────────────────────────────────────────

def validate_and_score(
    wav_path: Path,
    reference_text: Optional[str],
    transcriber: WhisperTranscriber,
    cfg: dict,
) -> Tuple[bool, dict]:
    """
    Validate one WAV file and return (is_valid, stats_dict).

    is_valid=False means the file is rejected (not added to dataset).
    """
    sr      = cfg["sample_rate"]
    min_dur = cfg["min_duration_s"]
    max_dur = cfg["max_duration_s"]
    thresh  = cfg["mismatch_threshold"]
    norm_cfg = cfg["text_normalization"]

    stats = {
        "filename": wav_path.name,
        "text":     reference_text or "",
        "duration": 0.0,
        "rms_db":   0.0,
        "snr_db":   0.0,
        "wer":      None,
        "flagged":  False,
        "reject_reason": "",
    }

    # 1. Load audio
    try:
        audio, _ = load_audio(wav_path, target_sr=sr)
    except Exception as exc:
        stats["reject_reason"] = f"load_failed: {exc}"
        return False, stats

    dur = get_duration(audio, sr)
    stats["duration"] = round(dur, 3)
    stats["rms_db"]   = round(compute_rms_db(audio), 2)
    stats["snr_db"]   = round(compute_snr(audio, sr), 2)

    # 2. Duration filter
    if dur < min_dur:
        stats["reject_reason"] = f"too_short ({dur:.2f}s < {min_dur}s)"
        return False, stats
    if dur > max_dur:
        stats["reject_reason"] = f"too_long ({dur:.2f}s > {max_dur}s)"
        return False, stats

    # 3. Transcription + WER
    transcription = ""
    try:
        transcription = transcriber.transcribe(wav_path)
    except Exception as exc:
        logger.warning(f"  Whisper failed on {wav_path.name}: {exc}")

    if reference_text and transcription:
        wer_val = compute_wer(reference_text, transcription)
        stats["wer"] = round(wer_val, 4)
        if wer_val > thresh:
            stats["flagged"] = True
            logger.warning(
                f"  FLAGGED {wav_path.name}: WER={wer_val:.2f}  "
                f"ref='{reference_text[:60]}…'  hyp='{transcription[:60]}…'"
            )
    elif not reference_text and transcription:
        # Use transcription as the text
        stats["text"] = transcription

    # 4. Text normalisation
    raw_text  = stats["text"]
    norm_text = normalize_text(
        raw_text,
        lowercase=norm_cfg.get("lowercase", True),
        remove_special_chars=norm_cfg.get("remove_special_chars", True),
        expand_abbrevs=norm_cfg.get("expand_abbreviations", True),
    )

    if not norm_text:
        stats["reject_reason"] = "empty_text_after_normalisation"
        return False, stats

    stats["text"]      = raw_text
    stats["norm_text"] = norm_text
    return True, stats


def split_dataset(
    valid_items: List[dict],
    train_r: float,
    val_r: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Stratified random split into train / val / test."""
    random.seed(seed)
    items = valid_items.copy()
    random.shuffle(items)
    n      = len(items)
    n_val  = max(1, round(n * val_r))
    n_test = max(1, round(n * (1.0 - train_r - val_r)))
    n_train = n - n_val - n_test
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


# ── Output writers ─────────────────────────────────────────────────────────

def write_ljspeech_metadata(items: List[dict], path: Path) -> None:
    """
    Write LJSpeech-format metadata.csv:
      filename (no ext)|raw text|normalized text
    No header row, pipe-delimited.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        for item in items:
            stem      = Path(item["filename"]).stem
            raw_text  = item.get("text", "").replace("|", " ")
            norm_text = item.get("norm_text", raw_text).replace("|", " ")
            f.write(f"{stem}|{raw_text}|{norm_text}\n")


def write_split_txt(items: List[dict], path: Path) -> None:
    """Write one filename stem per line (for train.txt / val.txt / test.txt)."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(Path(item["filename"]).stem + "\n")


def write_stats_csv(all_stats: List[dict], path: Path) -> None:
    if not all_stats:
        return
    fieldnames = ["filename", "duration", "rms_db", "snr_db", "wer", "flagged", "reject_reason", "text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_stats)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 4: Dataset Preparation (LJSpeech format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",        "-i", required=True,
                   help="Folder with WAV files (+ optional metadata.csv).")
    p.add_argument("--output",       "-o", required=True,
                   help="Output dataset folder.")
    p.add_argument("--config",       "-c", default=None)
    p.add_argument("--log-dir",            default="./logs")
    p.add_argument("--metadata",           default=None,
                   help="Path to metadata.csv (default: <input>/metadata.csv).")
    p.add_argument("--no-metadata",  action="store_true",
                   help="Ignore any metadata.csv; use Whisper for all texts.")
    p.add_argument("--no-transcribe", action="store_true",
                   help="Skip Whisper transcription (no WER check).")
    p.add_argument("--whisper-model", default=None,
                   help="Whisper model size override.")
    p.add_argument("--copy-wavs",    action="store_true",
                   help="Copy WAVs to output/dataset/wavs/ (default: symlink).")
    return p.parse_args()


def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f)
        if "stage4" in full:
            def _merge(base: dict, override: dict) -> dict:
                out = base.copy()
                for k, v in override.items():
                    if isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k] = {**out[k], **v}
                    else:
                        out[k] = v
                return out
            cfg = _merge(cfg, full["stage4"])
        if "sample_rate" in full:
            cfg.setdefault("sample_rate", full["sample_rate"])
    return cfg


def main() -> None:
    global logger
    args   = parse_args()
    logger = setup_logger("stage4_prepare", log_dir=args.log_dir)

    cfg = load_config(args.config)
    if args.whisper_model:
        cfg["whisper_model"] = args.whisper_model

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Collect WAV files
    wavs = sorted(input_dir.glob("*.wav"))
    if not wavs:
        logger.error(f"No WAV files found in {input_dir}")
        sys.exit(1)
    logger.info(f"Found {len(wavs)} WAV file(s).")

    # Load existing metadata
    meta_path = Path(args.metadata) if args.metadata else input_dir / "metadata.csv"
    if args.no_metadata:
        reference_map: Dict[str, str] = {}
        logger.info("--no-metadata: will use Whisper transcription as text.")
    else:
        reference_map = load_metadata(meta_path)
        logger.info(
            f"Loaded {len(reference_map)} reference text(s) from "
            f"{meta_path.name if meta_path.exists() else '(not found)'}"
        )

    # Initialise Whisper (lazy)
    transcriber = WhisperTranscriber(
        model_size = cfg["whisper_model"],
        language   = cfg["whisper_language"],
        device     = cfg["whisper_device"],
    ) if not args.no_transcribe else None

    # Process each WAV
    valid_items: List[dict] = []
    all_stats:   List[dict] = []
    rejected_names: List[str] = []

    try:
        from tqdm import tqdm
        wav_iter = tqdm(wavs, desc="Validating", unit="file")
    except ImportError:
        wav_iter = wavs

    for wav_path in wav_iter:
        stem      = wav_path.stem
        ref_text  = reference_map.get(stem, None)

        if args.no_transcribe:
            # Skip transcription — just check duration + normalise text
            try:
                audio, _ = load_audio(wav_path, target_sr=cfg["sample_rate"])
            except Exception as exc:
                logger.warning(f"Skipping {wav_path.name}: {exc}")
                all_stats.append({"filename": wav_path.name, "reject_reason": str(exc)})
                rejected_names.append(wav_path.name)
                continue

            dur = get_duration(audio, cfg["sample_rate"])
            if dur < cfg["min_duration_s"] or dur > cfg["max_duration_s"]:
                reason = f"duration {dur:.2f}s out of range"
                logger.warning(f"Rejecting {wav_path.name}: {reason}")
                all_stats.append({"filename": wav_path.name, "reject_reason": reason,
                                   "duration": dur})
                rejected_names.append(wav_path.name)
                continue

            norm_cfg = cfg["text_normalization"]
            raw_text  = ref_text or ""
            norm_text = normalize_text(
                raw_text,
                lowercase=norm_cfg.get("lowercase", True),
                remove_special_chars=norm_cfg.get("remove_special_chars", True),
                expand_abbrevs=norm_cfg.get("expand_abbreviations", True),
            )

            item = {
                "filename":  wav_path.name,
                "text":      raw_text,
                "norm_text": norm_text,
                "duration":  round(dur, 3),
                "rms_db":    round(compute_rms_db(audio), 2),
                "snr_db":    round(compute_snr(audio, cfg["sample_rate"]), 2),
                "flagged":   False,
                "wer":       None,
            }
            all_stats.append(item)
            valid_items.append(item)

        else:
            is_valid, stats = validate_and_score(
                wav_path, ref_text, transcriber, cfg
            )
            all_stats.append(stats)
            if is_valid:
                valid_items.append(stats)
            else:
                logger.warning(f"Rejected {wav_path.name}: {stats['reject_reason']}")
                rejected_names.append(wav_path.name)

    logger.info(
        f"Validated: {len(valid_items)} accepted, {len(rejected_names)} rejected."
    )

    if not valid_items:
        logger.error("No valid files to build a dataset from. Exiting.")
        sys.exit(1)

    # Split
    train_items, val_items, test_items = split_dataset(
        valid_items,
        train_r = cfg["train_ratio"],
        val_r   = cfg["val_ratio"],
        seed    = cfg["random_seed"],
    )
    logger.info(
        f"Split → train: {len(train_items)} | val: {len(val_items)} | test: {len(test_items)}"
    )

    # Build output directory structure
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Copy or symlink WAVs
    for item in valid_items:
        src = input_dir / item["filename"]
        dst = wavs_dir / item["filename"]
        if dst.exists():
            continue
        try:
            if args.copy_wavs:
                shutil.copy2(src, dst)
            else:
                # Use relative symlink when possible
                try:
                    rel = src.resolve().relative_to(wavs_dir.resolve())
                    dst.symlink_to(rel)
                except (ValueError, OSError):
                    dst.symlink_to(src.resolve())
        except Exception as exc:
            logger.warning(f"Could not link/copy {src.name}: {exc}")

    # LJSpeech metadata.csv (all items)
    write_ljspeech_metadata(valid_items, output_dir / "metadata.csv")

    # Split manifests
    write_split_txt(train_items, output_dir / "train.txt")
    write_split_txt(val_items,   output_dir / "val.txt")
    write_split_txt(test_items,  output_dir / "test.txt")

    # Stats CSV
    write_stats_csv(all_stats, output_dir / "stats.csv")

    # Rejected list
    if rejected_names:
        (output_dir / "rejected.txt").write_text(
            "\n".join(rejected_names), encoding="utf-8"
        )

    # Summary JSON (handy for run_pipeline.py)
    summary = {
        "total_files":    len(wavs),
        "valid_files":    len(valid_items),
        "rejected_files": len(rejected_names),
        "flagged_files":  sum(1 for s in all_stats if s.get("flagged")),
        "total_duration_s": round(sum(s.get("duration", 0) for s in valid_items), 2),
        "train": len(train_items),
        "val":   len(val_items),
        "test":  len(test_items),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info(f"Stage 4 complete.")
    logger.info(f"  Total duration : {summary['total_duration_s']:.1f}s "
                f"({summary['total_duration_s']/3600:.2f}h)")
    logger.info(f"  Flagged (WER)  : {summary['flagged_files']}")
    logger.info(f"  Rejected       : {summary['rejected_files']}")
    logger.info(f"  Output         : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
