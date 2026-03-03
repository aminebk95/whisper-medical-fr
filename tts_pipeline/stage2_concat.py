#!/usr/bin/env python3
"""
Stage 2 — Audio Concatenation
==============================
Concatenates cleaned audio clips into longer files, with configurable
silence padding, fade in/out, and automatic splitting by max duration.

Accepts an optional manifest file (one filename per line) to control
clip order; otherwise processes clips in alphabetical order.

CLI usage:
    python stage2_concat.py --input ./output/cleaned --output ./output/concatenated
    python stage2_concat.py --input ./output/cleaned --output ./output/concatenated --manifest order.txt
    python stage2_concat.py --input ./output/cleaned --output ./output/concatenated --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_utils import setup_logger
from utils.audio_utils import load_audio, save_audio, get_duration

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "sample_rate": 22050,
    "silence_between_clips_ms": 500,
    "max_output_duration_s": 300,
    "fade_in_ms": 10,
    "fade_out_ms": 10,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def make_silence(duration_ms: int, sr: int) -> np.ndarray:
    """Return a zero-filled numpy array of *duration_ms* milliseconds."""
    return np.zeros(int(sr * duration_ms / 1000), dtype=np.float32)


def apply_fade(audio: np.ndarray, sr: int, fade_in_ms: int, fade_out_ms: int) -> np.ndarray:
    """Apply linear fade-in and fade-out to *audio*."""
    audio = audio.copy()
    fade_in_samples  = min(int(sr * fade_in_ms / 1000), len(audio) // 2)
    fade_out_samples = min(int(sr * fade_out_ms / 1000), len(audio) // 2)

    if fade_in_samples > 0:
        ramp_in = np.linspace(0, 1, fade_in_samples, dtype=np.float32)
        audio[:fade_in_samples] *= ramp_in

    if fade_out_samples > 0:
        ramp_out = np.linspace(1, 0, fade_out_samples, dtype=np.float32)
        audio[-fade_out_samples:] *= ramp_out

    return audio


def collect_clips(input_dir: Path, manifest: Path | None) -> List[Path]:
    """
    Return ordered list of WAV files.

    If *manifest* is given, each line is a filename (optionally relative
    to *input_dir*).  Otherwise, files are sorted alphabetically.
    """
    if manifest and manifest.exists():
        clips = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = input_dir / p
            if p.exists():
                clips.append(p)
            else:
                print(f"[WARN] Manifest entry not found, skipping: {p}")
        return clips

    return sorted(input_dir.glob("*.wav"))


def flush_buffer(
    buffer: List[np.ndarray],
    output_dir: Path,
    file_index: int,
    sr: int,
    fade_in_ms: int,
    fade_out_ms: int,
    logger,
) -> None:
    """Concatenate *buffer* and write to disk."""
    if not buffer:
        return
    combined = np.concatenate(buffer)
    combined = apply_fade(combined, sr, fade_in_ms, fade_out_ms)
    out_path = output_dir / f"concat_{file_index:04d}.wav"
    save_audio(combined, out_path, sr=sr)
    dur = get_duration(combined, sr)
    logger.info(f"  Saved {out_path.name}  ({dur:.1f}s, {len(buffer)} clips)")


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2: Audio Concatenation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",    "-i", required=True, help="Folder of cleaned WAV clips.")
    p.add_argument("--output",   "-o", required=True, help="Folder for concatenated output.")
    p.add_argument("--manifest", "-m", default=None,  help="Optional text file listing clip order.")
    p.add_argument("--config",   "-c", default=None,  help="Path to config.yaml.")
    p.add_argument("--log-dir",        default="./logs")
    p.add_argument("--silence-ms", type=int, default=None,
                   help="Override silence between clips (ms).")
    p.add_argument("--max-duration", type=float, default=None,
                   help="Override max output file duration (seconds).")
    return p.parse_args()


def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f)
        if "stage2" in full:
            cfg.update(full["stage2"])
        if "sample_rate" in full:
            cfg["sample_rate"] = full["sample_rate"]
    return cfg


def main() -> None:
    args = parse_args()
    logger = setup_logger("stage2_concat", log_dir=args.log_dir)

    cfg = load_config(args.config)
    if args.silence_ms is not None:
        cfg["silence_between_clips_ms"] = args.silence_ms
    if args.max_duration is not None:
        cfg["max_output_duration_s"] = args.max_duration

    sr          = cfg["sample_rate"]
    silence_ms  = cfg["silence_between_clips_ms"]
    max_dur_s   = cfg["max_output_duration_s"]
    fade_in_ms  = cfg.get("fade_in_ms", 10)
    fade_out_ms = cfg.get("fade_out_ms", 10)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest) if args.manifest else None
    clips    = collect_clips(input_dir, manifest)

    if not clips:
        logger.error(f"No WAV files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(clips)} clip(s). Max output duration: {max_dur_s}s.")

    silence_pad   = make_silence(silence_ms, sr)
    buffer: List[np.ndarray] = []
    buffer_dur    = 0.0
    file_index    = 1
    total_clips   = 0
    failed_clips  = 0

    try:
        from tqdm import tqdm
        clip_iter = tqdm(clips, desc="Concatenating", unit="clip")
    except ImportError:
        clip_iter = clips

    for clip_path in clip_iter:
        try:
            audio, _ = load_audio(clip_path, target_sr=sr)
        except Exception as exc:
            logger.warning(f"Could not load {clip_path.name}: {exc} — skipping.")
            failed_clips += 1
            continue

        clip_dur = get_duration(audio, sr)

        # If adding this clip would exceed the limit, flush first
        if buffer and (buffer_dur + clip_dur) > max_dur_s:
            flush_buffer(buffer, output_dir, file_index, sr, fade_in_ms, fade_out_ms, logger)
            buffer      = []
            buffer_dur  = 0.0
            file_index += 1

        # Append silence separator (not before the very first clip)
        if buffer:
            buffer.append(silence_pad.copy())
            buffer_dur += silence_ms / 1000

        buffer.append(audio)
        buffer_dur += clip_dur
        total_clips += 1

    # Flush remaining buffer
    flush_buffer(buffer, output_dir, file_index, sr, fade_in_ms, fade_out_ms, logger)

    logger.info("=" * 60)
    logger.info(
        f"Stage 2 complete. {total_clips} clips concatenated into "
        f"{file_index} file(s). {failed_clips} failed."
    )
    logger.info(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
